#!/bin/bash

echo ""
echo "========================================"
echo "  机器学习学习之旅 - 一键安装"
echo "========================================"
echo ""

# 检查 Python 版本
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python3，请先安装 Python 3.10+"
    exit 1
fi

echo "[1/4] 检测到 Python:"
python3 --version
echo ""

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "[2/4] 创建虚拟环境..."
    python3 -m venv venv
else
    echo "[2/4] 虚拟环境已存在，跳过创建"
fi
echo ""

# 激活虚拟环境并安装依赖
echo "[3/4] 安装依赖包（首次可能需要几分钟）..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

if [ $? -ne 0 ]; then
    echo "[错误] 依赖安装失败，请检查网络连接"
    exit 1
fi
echo ""

# 验证安装
echo "[4/4] 验证安装..."
python3 -c "import cv2; import openvino; import ultralytics; print('OpenCV:', cv2.__version__); print('OpenVINO: OK'); print('Ultralytics: OK')"
echo ""

echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "使用方法:"
echo "  1. 激活环境: source venv/bin/activate"
echo "  2. 运行目标检测: python 02-computer-vision/object-detection/screen_simple.py"
echo "  3. 运行姿态估计: python 02-computer-vision/pose-estimation/pose_app.pyw"
echo "  4. 运行人脸识别: python 02-computer-vision/face-recognition/face_app.pyw"
echo ""
