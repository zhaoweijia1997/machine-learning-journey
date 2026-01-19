"""
环境测试脚本
验证所有依赖库是否正确安装

作者: zhaoweijia1997
日期: 2026-01-19
"""

import sys

def test_imports():
    """测试所有必需的库"""
    print("=" * 60)
    print("🧪 开始测试环境配置...")
    print("=" * 60)
    print()

    tests = [
        ("Python 版本", lambda: sys.version, None),
        ("NumPy", lambda: __import__('numpy').__version__, 'numpy'),
        ("Pandas", lambda: __import__('pandas').__version__, 'pandas'),
        ("Matplotlib", lambda: __import__('matplotlib').__version__, 'matplotlib'),
        ("OpenCV", lambda: __import__('cv2').__version__, 'cv2'),
        ("Pillow", lambda: __import__('PIL').__version__, 'PIL'),
        ("PyTorch", lambda: __import__('torch').__version__, 'torch'),
        ("TorchVision", lambda: __import__('torchvision').__version__, 'torchvision'),
        ("Ultralytics (YOLO)", lambda: __import__('ultralytics').__version__, 'ultralytics'),
    ]

    passed = 0
    failed = 0

    for name, test_func, module_name in tests:
        try:
            if module_name:
                version = test_func()
                print(f"✅ {name:25s} {version}")
            else:
                info = test_func()
                print(f"✅ {name:25s} {info.split()[0]}")
            passed += 1
        except ImportError as e:
            print(f"❌ {name:25s} 未安装")
            failed += 1
        except Exception as e:
            print(f"⚠️  {name:25s} 错误: {str(e)}")
            failed += 1

    print()
    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0

def test_torch_device():
    """测试 PyTorch 设备"""
    print()
    print("🔧 检测 PyTorch 设备...")
    print("-" * 60)

    try:
        import torch

        print(f"PyTorch 版本: {torch.__version__}")

        # 检查 CPU
        print(f"✅ CPU 可用")

        # 检查 CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用")
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
        else:
            print(f"ℹ️  CUDA 不可用 (未检测到 NVIDIA GPU)")

        # 检查 MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"✅ MPS 可用 (Apple Silicon)")
        else:
            print(f"ℹ️  MPS 不可用 (非 Mac M 系列芯片)")

        # 检查 XPU (Intel GPU)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print(f"✅ XPU 可用 (Intel GPU)")
            print(f"   设备数量: {torch.xpu.device_count()}")
            print(f"   设备名称: {torch.xpu.get_device_name(0)}")
        else:
            print(f"ℹ️  XPU 不可用")
            print(f"   你的 Intel Ultra 9 185H 集成了 Intel Arc GPU")
            print(f"   如需 GPU 加速，可以安装 intel-extension-for-pytorch:")
            print(f"   pip install intel-extension-for-pytorch")

        # 测试基本运算
        print()
        print("🧮 测试基本运算...")
        x = torch.rand(3, 3)
        y = torch.rand(3, 3)
        z = x + y
        print(f"✅ PyTorch 运算测试通过")

    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

    print("-" * 60)
    return True

def test_yolo():
    """测试 YOLO 模型"""
    print()
    print("🎯 测试 YOLOv8...")
    print("-" * 60)

    try:
        from ultralytics import YOLO
        import numpy as np

        print("正在下载 YOLOv8n 模型（首次运行需要下载）...")
        model = YOLO('yolov8n.pt')

        print("✅ 模型加载成功")

        # 测试推理
        print("测试模型推理...")
        # 创建一个随机图片
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_img, verbose=False)

        print(f"✅ 推理测试通过")
        print(f"   模型类别数: {len(model.names)}")
        print(f"   人(person)类别ID: 0")

    except Exception as e:
        print(f"❌ YOLO 测试失败: {e}")
        return False

    print("-" * 60)
    return True

def main():
    """主函数"""
    print()
    print("🔬 机器学习环境配置测试")
    print()

    # 测试导入
    if not test_imports():
        print()
        print("⚠️  部分库未安装，请运行:")
        print("   pip install -r 00-environment/requirements.txt")
        return

    # 测试 PyTorch
    if not test_torch_device():
        print()
        print("⚠️  PyTorch 设备测试失败")
        return

    # 测试 YOLO
    if not test_yolo():
        print()
        print("⚠️  YOLO 测试失败")
        return

    # 全部通过
    print()
    print("=" * 60)
    print("🎉 恭喜！所有测试通过，环境配置成功！")
    print("=" * 60)
    print()
    print("📚 下一步:")
    print("   1. 运行基础检测: python detect_person_basic.py")
    print("   2. 实时摄像头检测: python detect_person_webcam.py")
    print()
    print("💡 提示:")
    print("   - 首次运行会自动下载 YOLOv8 模型")
    print("   - 模型文件约 6MB，会保存在当前目录")
    print("   - 详细文档请查看 README.md")
    print()

if __name__ == "__main__":
    main()
