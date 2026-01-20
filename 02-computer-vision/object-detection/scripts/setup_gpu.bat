@echo off
REM GPU 加速一键配置脚本

echo ======================================
echo GPU 加速环境配置
echo ======================================
echo.

echo [1/3] 激活虚拟环境...
call ..\..\venv\Scripts\activate

echo.
echo [2/3] 检查 OpenVINO 安装...
python -c "import openvino; print('OpenVINO 版本:', openvino.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo OpenVINO 未安装，正在安装...
    pip install openvino openvino-dev
) else (
    echo OpenVINO 已安装
)

echo.
echo [3/3] 转换模型为 OpenVINO 格式...
if exist yolov8n_openvino_model (
    echo 模型已转换
) else (
    echo 正在转换模型（需要 1-2 分钟）...
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='openvino')"
    echo 转换完成！
)

echo.
echo ======================================
echo 配置完成！
echo ======================================
echo.
echo 现在可以运行:
echo   python detect_gpu.py         # GPU 图片检测
echo   python webcam_gpu.py         # GPU 实时检测
echo   python benchmark.py          # 性能测试
echo.
pause
