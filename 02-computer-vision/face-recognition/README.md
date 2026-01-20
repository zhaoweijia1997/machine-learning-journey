# 人脸识别系统

基于 OpenVINO 的人脸识别系统，支持人脸录入和实时识别。

## 功能

- 人脸检测：使用 OpenVINO face-detection-retail-0005 模型
- 人脸识别：使用 OpenVINO face-reidentification-retail-0095 模型
- 支持摄像头实时识别
- 支持屏幕捕获识别
- GUI 界面管理人脸数据库

## 使用方法

### 启动应用

双击 `启动人脸识别.bat` 或运行：

```bash
python face_app.pyw
```

### 录入人脸

1. 点击"摄像头"按钮启动摄像头
2. 点击"开始录入"输入姓名和职称
3. 面向摄像头，按 S 键拍照录入
4. 建议录入多张不同角度的照片

### 识别人脸

1. 选择"摄像头"或"屏幕捕获"模式
2. 系统自动识别画面中的人脸
3. 已录入的人脸会显示姓名和置信度

## 文件说明

- `face_app.pyw` - GUI 应用主程序
- `face_register.py` - 人脸数据库管理模块
- `face_db/` - 人脸数据库目录
  - `embeddings.pkl` - 人脸特征向量
  - `face_info.json` - 人脸信息
  - `photos/` - 录入的照片
  - `models/` - OpenVINO 模型文件

## 依赖

- Python 3.10+
- OpenVINO
- OpenCV
- NumPy
- Pillow
- dxcam (可选，屏幕捕获加速)
- mss (可选，屏幕捕获备选)

## 快捷键

- `S` - 拍照录入
- `ESC` - 停止
