# 环境配置指南

## 方式一：使用 Conda（推荐）

### 1. 安装 Miniconda
下载并安装 Miniconda: https://docs.conda.io/en/latest/miniconda.html

### 2. 创建虚拟环境
```bash
conda env create -f environment.yml
```

### 3. 激活环境
```bash
conda activate ml-cv
```

### 4. 验证安装
```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

## 方式二：使用 pip + venv

### 1. 创建虚拟环境
```bash
python -m venv venv
```

### 2. 激活虚拟环境

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

## Intel Arc GPU 优化（可选但推荐）

### 安装 Intel Extension for PyTorch
```bash
pip install intel-extension-for-pytorch
```

### 安装 OpenVINO
访问 Intel OpenVINO 官网下载适合你系统的版本:
https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html

Windows 快速安装:
```bash
pip install openvino openvino-dev
```

### 验证 Intel Arc GPU
```python
import torch
import intel_extension_for_pytorch as ipex

# 检查 XPU (Intel GPU) 是否可用
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f"Intel Arc GPU 可用！设备数量: {torch.xpu.device_count()}")
    print(f"设备名称: {torch.xpu.get_device_name(0)}")
else:
    print("未检测到 Intel Arc GPU，将使用 CPU")
```

## 常用命令

### 查看已安装的包
```bash
conda list
# 或
pip list
```

### 更新包
```bash
conda update --all
# 或
pip install --upgrade package_name
```

### 导出当前环境
```bash
conda env export > environment_backup.yml
# 或
pip freeze > requirements_backup.txt
```

## 故障排除

### PyTorch GPU 支持
如果需要 CUDA 支持（NVIDIA GPU），访问:
https://pytorch.org/get-started/locally/

### OpenCV 报错
如果 opencv 出现 DLL 加载错误，尝试:
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-headless
```

### Jupyter Notebook 无法找到内核
```bash
python -m ipykernel install --user --name=ml-cv
```

## 开发工具推荐

- **IDE**: VS Code + Python 扩展
- **Jupyter**: JupyterLab 或 VS Code Jupyter 扩展
- **终端**: Windows Terminal (Windows) 或默认终端
