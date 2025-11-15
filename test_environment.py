"""
环境测试脚本

用于验证FOMAML训练所需的环境是否正确配置

使用方法:
    python test_environment.py
"""

import sys
import os


def test_python_version():
    """测试Python版本"""
    print("\n" + "=" * 60)
    print("测试1: Python版本")
    print("=" * 60)

    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print("✅ Python版本符合要求 (>= 3.8)")
        return True
    else:
        print("❌ Python版本过低，需要 >= 3.8")
        return False


def test_pytorch():
    """测试PyTorch"""
    print("\n" + "=" * 60)
    print("测试2: PyTorch")
    print("=" * 60)

    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")

        # 测试CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA可用")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"           显存: {memory_gb:.1f} GB")
        else:
            print("⚠️  CUDA不可用，将使用CPU（不推荐）")

        return True

    except ImportError:
        print("❌ PyTorch未安装")
        print("   安装命令: pip install torch")
        return False


def test_packages():
    """测试必需的包"""
    print("\n" + "=" * 60)
    print("测试3: 必需的Python包")
    print("=" * 60)

    required_packages = {
        'transformers': '4.35.0',
        'datasets': '2.14.0',
        'pandas': '2.0.0',
        'numpy': None,
        'tensordict': None,
        'tqdm': None,
        'omegaconf': None,
    }

    optional_packages = {
        'wandb': None,
    }

    all_ok = True

    # 检查必需包
    for package, min_version in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package:20s} {version}")
        except ImportError:
            print(f"❌ {package:20s} 未安装")
            all_ok = False

    # 检查可选包
    print("\n可选包:")
    for package, min_version in optional_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package:20s} {version}")
        except ImportError:
            print(f"⚠️  {package:20s} 未安装（可选，用于日志记录）")

    return all_ok


def test_verl():
    """测试verl框架"""
    print("\n" + "=" * 60)
    print("测试4: verl框架")
    print("=" * 60)

    try:
        # 尝试导入verl的核心模块
        from verl.utils.dataset import SFTDataset
        from verl.utils.tracking import Tracking
        print("✅ verl框架已正确安装")

        # 检查verl路径
        import verl
        verl_path = os.path.dirname(verl.__file__)
        print(f"   verl路径: {verl_path}")

        return True

    except ImportError as e:
        print(f"❌ verl框架导入失败: {e}")
        print("\n   安装verl:")
        print("   cd verl")
        print("   pip install -e .")
        return False


def test_model_path():
    """测试模型路径（可选）"""
    print("\n" + "=" * 60)
    print("测试5: 模型路径（可选）")
    print("=" * 60)

    model_path = "./models/Qwen3-4B-Instruct-2507"

    if os.path.exists(model_path):
        print(f"✅ 找到模型: {model_path}")

        # 检查关键文件
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            print(f"   ✅ config.json 存在")
        else:
            print(f"   ⚠️  config.json 不存在")

        return True
    else:
        print(f"⚠️  模型路径不存在: {model_path}")
        print("   请在配置文件中设置正确的模型路径")
        print("   或跳过此测试")
        return None  # 可选测试


def test_data_path():
    """测试数据路径（可选）"""
    print("\n" + "=" * 60)
    print("测试6: 数据路径（可选）")
    print("=" * 60)

    data_path = "./data/math_meta/meta_train"

    if os.path.exists(data_path):
        print(f"✅ 找到数据目录: {data_path}")

        # 统计parquet文件
        parquet_files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
        print(f"   找到 {len(parquet_files)} 个parquet文件")

        return True
    else:
        print(f"⚠️  数据目录不存在: {data_path}")
        print("   请先运行: python prepare_math_data.py")
        return None  # 可选测试


def test_gpu_memory():
    """测试GPU显存"""
    print("\n" + "=" * 60)
    print("测试7: GPU显存检查")
    print("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("⚠️  CUDA不可用，跳过显存检查")
            return None

        total_memory_required = 70  # GB (4卡估计)

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3

            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   总显存: {memory_gb:.1f} GB")

            # 分配测试（小量）
            try:
                test_tensor = torch.randn(1000, 1000).cuda(i)
                del test_tensor
                torch.cuda.empty_cache()
                print(f"   ✅ 显存分配测试通过")
            except RuntimeError as e:
                print(f"   ❌ 显存分配失败: {e}")
                return False

        print(f"\n预估显存需求: {total_memory_required} GB (4卡配置)")
        print("如果显存不足，请参考配置文件中的优化建议")

        return True

    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("FOMAML环境测试")
    print("=" * 80)

    results = []

    # 必需测试
    results.append(("Python版本", test_python_version()))
    results.append(("PyTorch", test_pytorch()))
    results.append(("Python包", test_packages()))
    results.append(("verl框架", test_verl()))

    # 可选测试
    model_result = test_model_path()
    if model_result is not None:
        results.append(("模型路径", model_result))

    data_result = test_data_path()
    if data_result is not None:
        results.append(("数据路径", data_result))

    gpu_result = test_gpu_memory()
    if gpu_result is not None:
        results.append(("GPU显存", gpu_result))

    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    for name, result in results:
        if result:
            print(f"✅ {name}")
        else:
            print(f"❌ {name}")

    all_required_passed = all(result for name, result in results if name in [
        "Python版本", "PyTorch", "Python包", "verl框架"
    ])

    print("\n" + "=" * 80)
    if all_required_passed:
        print("✅ 所有必需测试通过！环境配置正确。")
        print("\n下一步:")
        print("  1. 准备数据: python prepare_math_data.py")
        print("  2. 修改配置: config_qwen3_4b_math.yaml")
        print("  3. 开始训练: run_fomaml_qwen3_math.bat")
    else:
        print("❌ 部分必需测试失败，请修复后再继续。")
        print("\n常见问题:")
        print("  - PyTorch未安装: pip install torch")
        print("  - 缺少包: pip install transformers datasets pandas")
        print("  - verl未安装: cd verl && pip install -e .")
    print("=" * 80)


if __name__ == "__main__":
    main()
