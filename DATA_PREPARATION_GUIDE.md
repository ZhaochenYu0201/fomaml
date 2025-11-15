# FOMAML-SFT 数据准备完整指南

## 📋 目录

1. [核心数据格式](#核心数据格式)
2. [必需字段说明](#必需字段说明)
3. [Support-Query Split构建](#support-query-split构建)
4. [实际操作步骤](#实际操作步骤)
5. [数据格式示例](#数据格式示例)
6. [验证和调试](#验证和调试)

---

## 🎯 核心数据格式

### 最终数据格式（verl兼容）

FOMAML-SFT训练需要的数据格式是**parquet文件**，包含以下列：

```python
{
    'prompt': str,      # 用户输入/问题（必需）
    'response': str,    # 模型回答（必需）
    'metadata': str,    # 元数据，JSON字符串（可选）
}
```

### 文件组织结构

```
data/
├── meta_train/                    # 元训练数据
│   ├── task1_support.parquet     # 任务1的support集
│   ├── task1_query.parquet       # 任务1的query集
│   ├── task2_support.parquet     # 任务2的support集
│   ├── task2_query.parquet       # 任务2的query集
│   └── ...
├── few_shot_eval/                 # Few-shot评估数据
│   ├── task1_test.parquet
│   ├── task1_5shot.parquet
│   ├── task1_10shot.parquet
│   └── ...
└── baseline_sft/                  # Baseline SFT数据（可选）
    └── mixed_train.parquet        # 所有任务混合
```

---

## 📝 必需字段说明

### 1. `prompt` 字段（必需）

**作用**：用户的输入，模型看到的问题

**格式要求**：
- 类型：字符串（str）
- 可以包含：问题描述、上下文、指令等
- 支持多轮对话（使用chat template）

**示例**：
```python
prompt = """请解决以下数学问题。请提供详细的解题步骤和最终答案。

问题：求解方程 2x + 5 = 13

请一步步推理并给出答案。"""
```

**重要提示**：
- ⚠️ Prompt部分在训练时会被**masked掉**（不计算loss）
- ✅ 只有response部分计算loss
- 这是verl SFT的标准做法

### 2. `response` 字段（必需）

**作用**：模型应该生成的回答

**格式要求**：
- 类型：字符串（str）
- 应包含完整的推理过程和答案
- 推荐使用Chain-of-Thought格式

**示例**：
```python
response = """让我来一步步解决这个问题：

步骤1：移项
2x + 5 = 13
2x = 13 - 5
2x = 8

步骤2：求解
x = 8 / 2
x = 4

因此，答案是 x = 4"""
```

**训练行为**：
- ✅ Response部分**计算loss**
- 模型学习生成这样的回答
- 使用masked cross-entropy loss

### 3. `metadata` 字段（可选但推荐）

**作用**：存储额外信息，用于分析和调试

**格式要求**：
- 类型：JSON字符串（str）
- 内容：任意键值对

**示例**：
```python
metadata = json.dumps({
    'source': 'MATH',           # 数据来源
    'subject': 'algebra',       # 学科/任务
    'level': 'Level 3',         # 难度
    'original_id': '12345',     # 原始数据ID
})
```

**推荐包含的字段**：
```python
{
    'source': str,      # 数据来源（MATH, GSM8K等）
    'subject': str,     # 任务/学科
    'level': str,       # 难度级别（可选）
    'task_id': str,     # 任务标识符
}
```

---

## 🔧 Support-Query Split构建

### 概念说明

FOMAML需要将每个任务的数据分为两部分：

```
任务数据
  ├── Support Set (支持集)
  │   └── 用于内循环适应（inner loop adaptation）
  │       模型在这些数据上快速微调K步
  │
  └── Query Set (查询集)
      └── 用于元损失计算（meta loss）
          评估适应后的模型性能
```

### 数据划分比例

#### 推荐配置（FOMAML全参数）

```python
support_ratio = 0.20-0.30    # 20-30% 用于support
query_ratio = 0.30-0.40      # 30-40% 用于query
test_ratio = 0.30-0.50       # 30-50% 保留用于评估

# 示例：如果某任务有1000个样本
support: 300 samples  (30%)  # 内循环适应
query:   400 samples  (40%)  # 元损失计算
test:    300 samples  (30%)  # Few-shot评估
```

#### 样本数量建议

| 任务类型 | Support样本数 | Query样本数 | 总样本数 |
|---------|--------------|------------|---------|
| 简单任务 | 200-300 | 300-450 | 600-900 |
| 中等任务 | 300-500 | 450-750 | 900-1500 |
| 复杂任务 | 500-800 | 750-1200 | 1500-2400 |

**注意**：
- Support不要太少（<200可能不够适应）
- Query要比Support多（确保元损失稳定）
- 总数不要太多（每个任务>3000样本可能过拟合）

### 划分策略

#### 策略1: 随机划分（推荐用于同质任务）

```python
import random

def split_data_random(examples, support_ratio=0.3, query_ratio=0.4):
    """随机划分数据"""
    random.shuffle(examples)

    n_total = len(examples)
    n_support = int(n_total * support_ratio)
    n_query = int(n_total * query_ratio)

    support = examples[:n_support]
    query = examples[n_support:n_support + n_query]
    test = examples[n_support + n_query:]

    return support, query, test
```

#### 策略2: 分层划分（推荐用于有难度级别的数据）

```python
def split_data_stratified(examples, support_ratio=0.3, query_ratio=0.4):
    """按难度分层划分，确保各集合难度分布一致"""
    from collections import defaultdict

    # 按难度分组
    by_level = defaultdict(list)
    for ex in examples:
        level = ex.get('level', 'unknown')
        by_level[level].append(ex)

    support, query, test = [], [], []

    # 对每个难度级别独立划分
    for level, level_examples in by_level.items():
        random.shuffle(level_examples)
        n = len(level_examples)
        n_support = int(n * support_ratio)
        n_query = int(n * query_ratio)

        support.extend(level_examples[:n_support])
        query.extend(level_examples[n_support:n_support + n_query])
        test.extend(level_examples[n_support + n_query:])

    return support, query, test
```

#### 策略3: 时间划分（推荐用于时序数据）

```python
def split_data_temporal(examples, support_ratio=0.3, query_ratio=0.4):
    """按时间顺序划分（不打乱）"""
    # 假设examples已按时间排序
    n_total = len(examples)
    n_support = int(n_total * support_ratio)
    n_query = int(n_total * query_ratio)

    support = examples[:n_support]
    query = examples[n_support:n_support + n_query]
    test = examples[n_support + n_query:]

    return support, query, test
```

### 重要原则

✅ **DO（推荐做法）**：
- Support和Query来自同一任务/分布
- Query比Support稍多（提供更稳定的元梯度）
- 保留足够的Test数据用于评估
- 确保数据质量（去重、清洗）

❌ **DON'T（避免）**：
- Support和Query数据泄露（重复样本）
- Support太少（<100样本可能不够）
- Query太少（<150样本元梯度不稳定）
- 所有任务的Support/Query比例差异过大

---

## 🚀 实际操作步骤

### 步骤1: 准备原始数据

#### 方案A: 从公开数据集（推荐用于快速开始）

```bash
# 使用现有脚本自动准备MATH、GSM8K、ScienceQA数据
python prepare_math_science_data.py \
    --output-dir ./data/math_science_meta \
    --support-ratio 0.30 \
    --query-ratio 0.40 \
    --seed 42

# 这会自动：
# 1. 下载数据集
# 2. 按任务分组
# 3. 生成support/query split
# 4. 保存为parquet格式
```

#### 方案B: 使用自己的数据

**原始数据格式（任意格式均可）**：

```python
# 示例：JSON格式
your_raw_data = [
    {
        "question": "什么是光合作用？",
        "answer": "光合作用是植物利用光能...",
        "category": "biology",
        "difficulty": "easy"
    },
    {
        "question": "计算圆的面积公式是什么？",
        "answer": "圆的面积公式是 A = πr²...",
        "category": "math",
        "difficulty": "medium"
    },
    # ... 更多数据
]

# 或CSV格式
# question,answer,category,difficulty
# "什么是光合作用？","光合作用是植物利用光能...","biology","easy"
# ...

# 或任何你喜欢的格式！
```

### 步骤2: 按任务分组

```python
from collections import defaultdict

def group_by_task(raw_data, task_key='category'):
    """按任务字段分组"""
    tasks = defaultdict(list)

    for item in raw_data:
        task_name = item.get(task_key, 'default')
        tasks[task_name].append(item)

    return tasks

# 使用
tasks = group_by_task(your_raw_data, task_key='category')

print(f"Found {len(tasks)} tasks:")
for task_name, examples in tasks.items():
    print(f"  {task_name}: {len(examples)} examples")
```

**输出示例**：
```
Found 3 tasks:
  biology: 450 examples
  math: 680 examples
  physics: 520 examples
```

### 步骤3: 格式转换

创建格式转换函数，将原始格式转为verl格式：

```python
import json

def format_example(raw_example):
    """
    将原始数据转换为verl SFT格式

    必须返回: {'prompt': str, 'response': str, 'metadata': str}
    """
    # 提取字段
    question = raw_example['question']
    answer = raw_example['answer']
    category = raw_example.get('category', 'unknown')
    difficulty = raw_example.get('difficulty', 'unknown')

    # 构建prompt（加入指令）
    prompt = f"""请回答以下问题，提供详细解释。

问题：{question}

请提供答案和解释。"""

    # 构建response
    response = answer

    # 构建metadata
    metadata = json.dumps({
        'source': 'my_dataset',
        'category': category,
        'difficulty': difficulty,
    })

    return {
        'prompt': prompt,
        'response': response,
        'metadata': metadata,
    }

# 测试
test_example = {
    'question': '什么是光合作用？',
    'answer': '光合作用是植物利用光能...',
    'category': 'biology',
    'difficulty': 'easy'
}

formatted = format_example(test_example)
print(formatted)
```

### 步骤4: 生成Support-Query Split并保存

```python
import pandas as pd
from pathlib import Path

def create_support_query_split(
    task_name,
    examples,
    output_dir,
    support_ratio=0.30,
    query_ratio=0.40,
    format_func=format_example
):
    """为单个任务创建support-query split"""
    import random

    # 1. 打乱数据
    random.shuffle(examples)

    # 2. 计算划分点
    n_total = len(examples)
    n_support = int(n_total * support_ratio)
    n_query = int(n_total * query_ratio)

    # 3. 划分
    support_examples = examples[:n_support]
    query_examples = examples[n_support:n_support + n_query]
    test_examples = examples[n_support + n_query:]

    print(f"\n{task_name}:")
    print(f"  Total: {n_total}")
    print(f"  Support: {len(support_examples)}")
    print(f"  Query: {len(query_examples)}")
    print(f"  Test: {len(test_examples)}")

    # 4. 格式转换
    support_data = [format_func(ex) for ex in support_examples]
    query_data = [format_func(ex) for ex in query_examples]
    test_data = [format_func(ex) for ex in test_examples]

    # 5. 转换为DataFrame
    support_df = pd.DataFrame(support_data)
    query_df = pd.DataFrame(query_data)
    test_df = pd.DataFrame(test_data)

    # 6. 保存为parquet
    output_dir = Path(output_dir)
    (output_dir / "meta_train").mkdir(parents=True, exist_ok=True)
    (output_dir / "few_shot_eval").mkdir(parents=True, exist_ok=True)

    support_path = output_dir / "meta_train" / f"{task_name}_support.parquet"
    query_path = output_dir / "meta_train" / f"{task_name}_query.parquet"
    test_path = output_dir / "few_shot_eval" / f"{task_name}_test.parquet"

    support_df.to_parquet(support_path, index=False)
    query_df.to_parquet(query_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"  ✅ Saved: {support_path}")
    print(f"  ✅ Saved: {query_path}")
    print(f"  ✅ Saved: {test_path}")

    return support_df, query_df, test_df

# 为所有任务生成数据
output_dir = "./data/my_meta_learning"

for task_name, task_examples in tasks.items():
    create_support_query_split(
        task_name=task_name,
        examples=task_examples,
        output_dir=output_dir,
        support_ratio=0.30,
        query_ratio=0.40,
        format_func=format_example
    )
```

### 步骤5: 创建配置文件

创建YAML配置文件，指定数据路径：

```yaml
# config_my_fomaml.yaml

model:
  partial_pretrain: "meta-llama/Llama-3.2-1B"
  use_fsdp: true
  enable_gradient_checkpointing: true

data:
  max_length: 2048
  prompt_key: "prompt"
  response_key: "response"

meta:
  use_fomaml: true

  inner_lr: 1e-4
  num_inner_steps: 5
  inner_batch_size: 4

  outer_lr: 3e-5
  meta_batch_size: 4
  query_batch_size: 4

  tasks:
    - name: "biology"
      support_files: ["./data/my_meta_learning/meta_train/biology_support.parquet"]
      query_files: ["./data/my_meta_learning/meta_train/biology_query.parquet"]
      support_max_samples: 300
      query_max_samples: 450

    - name: "math"
      support_files: ["./data/my_meta_learning/meta_train/math_support.parquet"]
      query_files: ["./data/my_meta_learning/meta_train/math_query.parquet"]
      support_max_samples: 300
      query_max_samples: 450

    - name: "physics"
      support_files: ["./data/my_meta_learning/meta_train/physics_support.parquet"]
      query_files: ["./data/my_meta_learning/meta_train/physics_query.parquet"]
      support_max_samples: 300
      query_max_samples: 450

trainer:
  total_steps: 5000
  save_freq: 500
  test_freq: 100
  project_name: "my-fomaml-experiment"
  experiment_name: "biology-math-physics"
  default_local_dir: "./checkpoints/my_fomaml"
```

---

## 📊 数据格式示例

### 示例1: 数学推理任务

**原始数据**：
```json
{
    "problem": "Solve for x: 2x + 5 = 13",
    "solution": "2x = 13 - 5 = 8, so x = 4",
    "level": "Level 2",
    "type": "Algebra"
}
```

**转换后（verl格式）**：
```python
{
    "prompt": """请解决以下数学问题。请提供详细的解题步骤和最终答案。

问题：Solve for x: 2x + 5 = 13

请一步步推理并给出答案。""",

    "response": """让我来一步步解决这个问题：

步骤1：移项
2x + 5 = 13
2x = 13 - 5
2x = 8

步骤2：求解
x = 8 / 2
x = 4

因此，答案是 x = 4""",

    "metadata": '{"source": "MATH", "level": "Level 2", "type": "Algebra"}'
}
```

### 示例2: 科学问答任务

**原始数据**：
```json
{
    "question": "What is photosynthesis?",
    "answer": "Photosynthesis is the process by which plants use sunlight to produce glucose from carbon dioxide and water.",
    "subject": "biology",
    "difficulty": "medium"
}
```

**转换后**：
```python
{
    "prompt": """Please answer the following science question with a detailed explanation.

Question: What is photosynthesis?

Please provide your answer with reasoning.""",

    "response": """Photosynthesis is the process by which plants use sunlight to produce glucose from carbon dioxide and water.

This process occurs in the chloroplasts of plant cells and involves two main stages:
1. Light-dependent reactions
2. Light-independent reactions (Calvin cycle)

The overall equation is: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂""",

    "metadata": '{"source": "biology_dataset", "subject": "biology", "difficulty": "medium"}'
}
```

### 示例3: 代码生成任务

**原始数据**：
```json
{
    "instruction": "Write a function to reverse a string",
    "input": "",
    "output": "def reverse_string(s):\n    return s[::-1]",
    "language": "python"
}
```

**转换后**：
```python
{
    "prompt": """请根据以下指令编写代码。

指令：Write a function to reverse a string

请提供完整的代码实现和简要说明。""",

    "response": """下面是实现代码：

```python
def reverse_string(s):
    return s[::-1]
```

说明：
- 使用Python的切片语法 [::-1] 可以反转字符串
- 时间复杂度: O(n)
- 空间复杂度: O(n)""",

    "metadata": '{"source": "code_dataset", "language": "python", "task": "string_manipulation"}'
}
```

---

## ✅ 验证和调试

### 验证脚本

创建验证脚本检查数据质量：

```python
# validate_data.py

import pandas as pd
from pathlib import Path
import json

def validate_parquet_file(file_path):
    """验证单个parquet文件"""
    print(f"\n=== Validating {file_path.name} ===")

    # 1. 加载数据
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Loaded successfully: {len(df)} rows")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False

    # 2. 检查必需列
    required_columns = ['prompt', 'response']
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ Missing required column: {col}")
            return False
        print(f"✅ Column '{col}' exists")

    # 3. 检查空值
    for col in required_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"⚠️  Column '{col}' has {null_count} null values")
        else:
            print(f"✅ Column '{col}' has no null values")

    # 4. 检查数据类型
    for col in required_columns:
        if df[col].dtype != 'object':
            print(f"⚠️  Column '{col}' type is {df[col].dtype}, expected string")

    # 5. 检查样本长度
    prompt_lens = df['prompt'].str.len()
    response_lens = df['response'].str.len()

    print(f"\nPrompt length stats:")
    print(f"  Min: {prompt_lens.min()}")
    print(f"  Max: {prompt_lens.max()}")
    print(f"  Mean: {prompt_lens.mean():.1f}")

    print(f"\nResponse length stats:")
    print(f"  Min: {response_lens.min()}")
    print(f"  Max: {response_lens.max()}")
    print(f"  Mean: {response_lens.mean():.1f}")

    # 6. 检查metadata格式（如果存在）
    if 'metadata' in df.columns:
        print(f"\nMetadata validation:")
        valid_json = 0
        for i, meta in enumerate(df['metadata'].head(10)):
            try:
                json.loads(meta)
                valid_json += 1
            except:
                print(f"  ⚠️  Row {i}: Invalid JSON in metadata")
        print(f"  ✅ {valid_json}/10 samples have valid JSON metadata")

    # 7. 打印示例
    print(f"\n--- Sample (first row) ---")
    print(f"Prompt:\n{df['prompt'].iloc[0][:200]}...")
    print(f"\nResponse:\n{df['response'].iloc[0][:200]}...")

    return True

def validate_task(task_name, data_dir):
    """验证单个任务的support和query数据"""
    print(f"\n{'='*60}")
    print(f"Validating task: {task_name}")
    print(f"{'='*60}")

    data_dir = Path(data_dir)

    # 检查文件存在
    support_file = data_dir / "meta_train" / f"{task_name}_support.parquet"
    query_file = data_dir / "meta_train" / f"{task_name}_query.parquet"

    if not support_file.exists():
        print(f"❌ Support file not found: {support_file}")
        return False
    if not query_file.exists():
        print(f"❌ Query file not found: {query_file}")
        return False

    # 验证每个文件
    validate_parquet_file(support_file)
    validate_parquet_file(query_file)

    # 检查support和query的比例
    support_df = pd.read_parquet(support_file)
    query_df = pd.read_parquet(query_file)

    print(f"\n--- Split Statistics ---")
    print(f"Support samples: {len(support_df)}")
    print(f"Query samples: {len(query_df)}")
    print(f"Ratio (query/support): {len(query_df)/len(support_df):.2f}")

    if len(query_df) < len(support_df):
        print("⚠️  Warning: Query set is smaller than support set")
        print("   Recommended: Query >= Support")

    return True

# 使用
if __name__ == "__main__":
    data_dir = "./data/my_meta_learning"

    # 验证所有任务
    tasks = ["biology", "math", "physics"]

    for task in tasks:
        validate_task(task, data_dir)
```

### 运行验证

```bash
python validate_data.py
```

**期望输出**：
```
============================================================
Validating task: biology
============================================================

=== Validating biology_support.parquet ===
✅ Loaded successfully: 300 rows
✅ Column 'prompt' exists
✅ Column 'response' exists
✅ Column 'prompt' has no null values
✅ Column 'response' has no null values

Prompt length stats:
  Min: 120
  Max: 850
  Mean: 285.3

Response length stats:
  Min: 80
  Max: 650
  Mean: 220.5

--- Split Statistics ---
Support samples: 300
Query samples: 450
Ratio (query/support): 1.50
✅ All checks passed!
```

### 常见问题排查

#### 问题1: "Missing required column"

**原因**：parquet文件缺少必需的列

**解决**：
```python
# 检查你的format函数是否返回了所有必需字段
def format_example(ex):
    return {
        'prompt': ...,     # ✅ 必需
        'response': ...,   # ✅ 必需
        'metadata': ...,   # ⭕ 可选
    }
```

#### 问题2: "Query set is smaller than support set"

**原因**：Query比Support少，可能导致元梯度不稳定

**解决**：
```python
# 调整比例
create_support_query_split(
    support_ratio=0.25,  # 减少support
    query_ratio=0.45,    # 增加query
)
```

#### 问题3: Prompt或Response太短

**原因**：格式转换时丢失了信息

**解决**：
```python
# 确保prompt包含足够的上下文
prompt = f"""[指令]

问题：{question}

[要求答案格式]"""  # 添加更多上下文

# 确保response包含完整答案
response = f"""[推理过程]

{reasoning}

[最终答案]
{answer}"""
```

---

## 🎯 快速开始模板

### 完整的数据准备脚本模板

```python
# my_data_preparation.py

import json
import random
import pandas as pd
from pathlib import Path

# ============================================
# Step 1: 加载你的原始数据
# ============================================
def load_your_data():
    """加载你的原始数据（任意格式）"""
    # TODO: 替换为你的数据加载逻辑

    # 示例：从JSON加载
    # with open('your_data.json') as f:
    #     data = json.load(f)

    # 示例：从CSV加载
    # import pandas as pd
    # df = pd.read_csv('your_data.csv')
    # data = df.to_dict('records')

    # 返回格式：list of dicts
    return [
        {'question': '...', 'answer': '...', 'task': '...'},
        # ...
    ]

# ============================================
# Step 2: 按任务分组
# ============================================
def group_by_task(data, task_key='task'):
    """按任务分组"""
    from collections import defaultdict
    tasks = defaultdict(list)
    for item in data:
        task_name = item.get(task_key, 'default')
        tasks[task_name].append(item)
    return dict(tasks)

# ============================================
# Step 3: 格式转换函数
# ============================================
def format_example(raw_example):
    """
    将你的原始格式转换为verl格式

    TODO: 根据你的数据修改这个函数
    """
    # 提取字段（根据你的数据结构修改）
    question = raw_example['question']
    answer = raw_example['answer']

    # 构建prompt
    prompt = f"""请回答以下问题。

问题：{question}

请提供详细答案。"""

    # 构建response
    response = answer

    # 构建metadata
    metadata = json.dumps({
        'source': 'my_dataset',
        'task': raw_example.get('task', 'unknown'),
    })

    return {
        'prompt': prompt,
        'response': response,
        'metadata': metadata,
    }

# ============================================
# Step 4: Support-Query划分
# ============================================
def create_splits(task_name, examples, output_dir,
                 support_ratio=0.30, query_ratio=0.40):
    """创建support-query split"""
    random.shuffle(examples)

    n = len(examples)
    n_support = int(n * support_ratio)
    n_query = int(n * query_ratio)

    support = examples[:n_support]
    query = examples[n_support:n_support + n_query]
    test = examples[n_support + n_query:]

    # 格式转换
    support_data = [format_example(ex) for ex in support]
    query_data = [format_example(ex) for ex in query]
    test_data = [format_example(ex) for ex in test]

    # 保存
    output_dir = Path(output_dir)
    (output_dir / "meta_train").mkdir(parents=True, exist_ok=True)
    (output_dir / "few_shot_eval").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(support_data).to_parquet(
        output_dir / "meta_train" / f"{task_name}_support.parquet",
        index=False
    )
    pd.DataFrame(query_data).to_parquet(
        output_dir / "meta_train" / f"{task_name}_query.parquet",
        index=False
    )
    pd.DataFrame(test_data).to_parquet(
        output_dir / "few_shot_eval" / f"{task_name}_test.parquet",
        index=False
    )

    print(f"✅ {task_name}: support={len(support)}, query={len(query)}, test={len(test)}")

# ============================================
# Main: 完整流程
# ============================================
def main():
    # 配置
    output_dir = "./data/my_fomaml_data"
    support_ratio = 0.30
    query_ratio = 0.40

    # 1. 加载数据
    print("Loading data...")
    raw_data = load_your_data()
    print(f"Loaded {len(raw_data)} examples")

    # 2. 按任务分组
    print("\nGrouping by task...")
    tasks = group_by_task(raw_data, task_key='task')
    for task_name, examples in tasks.items():
        print(f"  {task_name}: {len(examples)} examples")

    # 3. 为每个任务创建splits
    print("\nCreating support-query splits...")
    for task_name, examples in tasks.items():
        create_splits(
            task_name=task_name,
            examples=examples,
            output_dir=output_dir,
            support_ratio=support_ratio,
            query_ratio=query_ratio,
        )

    print(f"\n✅ Done! Data saved to {output_dir}")
    print("\nNext steps:")
    print("1. Run validation: python validate_data.py")
    print("2. Update config YAML with data paths")
    print("3. Start training: python maml_sft_trainer.py")

if __name__ == "__main__":
    main()
```

**使用这个模板**：
1. 修改 `load_your_data()` 加载你的数据
2. 修改 `format_example()` 适配你的数据格式
3. 运行 `python my_data_preparation.py`

---

## 📚 参考资源

- **示例脚本**：`prepare_math_science_data.py`（完整实现）
- **配置示例**：`config_maml_sft_example.yaml`
- **验证脚本**：上面的`validate_data.py`

---

## ✅ 检查清单

准备数据前确认：

- [ ] 原始数据已准备好
- [ ] 每个任务至少有500+样本
- [ ] 数据已按任务分组
- [ ] 实现了format函数（原始格式→verl格式）
- [ ] 设置了合理的support/query比例
- [ ] 创建了输出目录
- [ ] 运行了数据验证脚本
- [ ] 更新了配置文件中的数据路径

---

**准备好数据后，就可以开始训练了！** 🚀

```bash
# 验证数据
python validate_data.py

# 开始训练
torchrun --nproc_per_node=4 maml_sft_trainer.py --config-name config_my_fomaml
```
