@echo off
REM 快速激活虚拟环境脚本

echo.
echo ====================================
echo  机器学习环境激活
echo ====================================
echo.

call venv\Scripts\activate

echo ✅ 虚拟环境已激活！
echo.
echo 💡 快速命令:
echo    cd 02-computer-vision\object-detection
echo    python test_setup.py          # 测试环境
echo    python detect_person_basic.py  # 基础检测
echo    python detect_person_webcam.py # 实时检测
echo.
