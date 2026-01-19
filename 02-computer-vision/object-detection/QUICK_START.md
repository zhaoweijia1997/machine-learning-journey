# 快速开始 - 人形检测

## 🚀 三分钟上手

### 1. 激活虚拟环境

**Windows CMD:**
```bash
venv\Scripts\activate
```

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Git Bash / Linux / Mac:**
```bash
source venv/bin/activate
```

###2. 测试环境

```bash
cd 02-computer-vision/object-detection
python test_setup.py
```

如果看到 "🎉 恭喜！所有测试通过"，说明环境配置成功！

### 3. 运行你的第一个人形检测

#### 方法 A：检测图片

```bash
python detect_person_basic.py
```

**注意**：首次运行会自动下载 YOLOv8n 模型（约 6MB）

如果你没有测试图片，可以在 Python 中下载示例：

```python
import urllib.request
urllib.request.urlretrieve('https://ultralytics.com/images/bus.jpg', 'test_image.jpg')
```

然后运行：
```bash
python detect_person_basic.py
```

#### 方法 B：实时摄像头检测（超酷！）

```bash
python detect_person_webcam.py
```

**按键说明**：
- `q` - 退出
- `s` - 保存当前帧
- `空格` - 暂停/继续

## 📝 完整示例

### 创建你的第一个检测脚本

创建文件 `my_first_detection.py`：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 检测图片
results = model('test_image.jpg')

# 显示结果
results[0].show()

# 保存结果
results[0].save('result.jpg')

print("检测完成！")
```

运行：
```bash
python my_first_detection.py
```

## 🎯 进阶使用

### 调整置信度阈值

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# 只显示置信度 > 70% 的检测结果
results = model('test.jpg', conf=0.7)
```

### 只检测人

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# 只检测类别 0 (person)
results = model('test.jpg', classes=[0])
```

### 批量处理

```python
from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt')

# 处理文件夹中的所有图片
image_folder = 'images/'
for img_file in os.listdir(image_folder):
    if img_file.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(image_folder, img_file)
        results = model(img_path)
        results[0].save(f'output_{img_file}')
        print(f"处理完成: {img_file}")
```

### 获取检测坐标

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model('test.jpg')

# 遍历所有检测到的对象
for box in results[0].boxes:
    # 获取类别
    class_id = int(box.cls[0])
    class_name = results[0].names[class_id]

    # 获取置信度
    confidence = float(box.conf[0])

    # 获取边界框坐标 (x1, y1, x2, y2)
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

    # 只处理人
    if class_id == 0:
        print(f"检测到人 - 置信度: {confidence:.2%}")
        print(f"  位置: ({int(x1)}, {int(y1)}) 到 ({int(x2)}, {int(y2)})")
```

## 🔧 常见问题

### Q: 模型下载失败
A: 手动下载模型文件：
```bash
# 从 GitHub releases 下载
# https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
```

### Q: 摄像头打不开
A:
1. 检查摄像头权限
2. 尝试修改 camera_id：
   ```python
   detect_webcam(camera_id=1)  # 尝试其他摄像头
   ```

### Q: 运行速度慢
A:
1. 使用更小的模型（yolov8n 是最快的）
2. 降低输入分辨率：
   ```python
   results = model('test.jpg', imgsz=320)  # 默认 640
   ```
3. 使用 Intel GPU 加速（需要额外配置 OpenVINO）

### Q: 想检测其他物体
A: YOLO 可以检测 80 种物体，类别列表：
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print(model.names)  # 显示所有类别
```

常见类别ID：
- 0: person (人)
- 1: bicycle (自行车)
- 2: car (汽车)
- 16: dog (狗)
- 17: cat (猫)
- ...

## 📚 下一步学习

1. ✅ **完成基础检测** - 你现在的位置
2. ⏭️ **学习姿态估计** - 检测人体关键点和姿态
3. ⏭️ **多目标跟踪** - 追踪视频中的多个对象
4. ⏭️ **模型训练** - 使用自己的数据训练模型

## 💡 有用的链接

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [COCO 数据集](https://cocodataset.org/)
- [项目 README](README.md)

## 🎉 开始你的旅程！

现在运行你的第一个检测：

```bash
cd 02-computer-vision/object-detection
python test_setup.py
python detect_person_basic.py
```

祝你学习愉快！🚀
