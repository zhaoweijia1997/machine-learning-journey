@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   机器学习学习之旅 - 一键安装
echo ========================================
echo.

REM 检查 Python 版本
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+
    echo 下载地址: https://python.org
    pause
    exit /b 1
)

echo [1/4] 检测到 Python:
python --version
echo.

REM 创建虚拟环境
if not exist "venv" (
    echo [2/4] 创建虚拟环境...
    python -m venv venv
) else (
    echo [2/4] 虚拟环境已存在，跳过创建
)
echo.

REM 激活虚拟环境并安装依赖
echo [3/4] 安装依赖包（首次可能需要几分钟）...
call venv\Scripts\activate.bat
pip install --upgrade pip -q
pip install -r requirements.txt -q

if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络连接
    pause
    exit /b 1
)
echo.

REM 验证安装
echo [4/4] 验证安装...
python -c "import cv2; import openvino; import ultralytics; print('OpenCV:', cv2.__version__); print('OpenVINO: OK'); print('Ultralytics: OK')"
echo.

echo ========================================
echo   安装完成！
echo ========================================
echo.
echo 使用方法:
echo   1. 激活环境: venv\Scripts\activate
echo   2. 运行目标检测: python 02-computer-vision\object-detection\screen_simple.py
echo   3. 运行姿态估计: python 02-computer-vision\pose-estimation\pose_app.pyw
echo   4. 运行人脸识别: python 02-computer-vision\face-recognition\face_app.pyw
echo.
pause
