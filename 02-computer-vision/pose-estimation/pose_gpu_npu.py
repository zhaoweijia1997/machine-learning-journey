# -*- coding: utf-8 -*-
"""
GPU + NPU 并行姿态估计
GPU 做主推理，NPU 做辅助任务
"""

import cv2
import numpy as np
import time
import os
import threading
from queue import Queue, Empty
from ultralytics import YOLO

# 屏幕捕获
try:
    import mss
    HAS_MSS = True
except:
    HAS_MSS = False


class ScreenCapture:
    def __init__(self, monitor_idx=0):
        self.monitor_idx = monitor_idx
        self.sct = mss.mss() if HAS_MSS else None
        self.monitor = None
        if self.sct:
            monitors = self.sct.monitors
            idx = monitor_idx + 1
            self.monitor = monitors[idx] if idx < len(monitors) else monitors[1]

    def grab(self):
        if self.sct and self.monitor:
            img = self.sct.grab(self.monitor)
            return np.array(img)[:, :, :3].copy()
        return None

    def release(self):
        if self.sct:
            self.sct.close()


class DualInference:
    """GPU + NPU 双设备推理"""

    def __init__(self):
        print("=" * 60)
        print("GPU + NPU 并行姿态估计")
        print("=" * 60)

        # 检查设备
        from openvino import Core
        ie = Core()
        devices = ie.available_devices
        print(f"可用设备: {devices}")

        self.has_npu = 'NPU' in devices
        self.has_gpu = 'GPU' in devices

        # 模型路径
        pose_model_path = 'yolov8n-pose_openvino_model'
        detect_model_path = '../object-detection/yolov8n_openvino_model'

        # 转换姿态模型
        if not os.path.exists(pose_model_path):
            print("转换姿态估计模型...")
            YOLO('yolov8n-pose.pt').export(format='openvino', half=False)

        # GPU: 姿态估计（主任务）
        print("\n加载 GPU 模型 (姿态估计)...")
        self.pose_model = YOLO(pose_model_path, task='pose')

        # NPU: 目标检测（辅助任务）
        if self.has_npu and os.path.exists(detect_model_path):
            print("加载 NPU 模型 (目标检测)...")
            # 使用 OpenVINO 直接加载到 NPU
            from openvino import Core
            self.ie = Core()
            model_xml = os.path.join(detect_model_path, 'yolov8n.xml')
            if os.path.exists(model_xml):
                self.npu_model = self.ie.compile_model(model_xml, 'NPU')
                self.npu_input = self.npu_model.input(0)
                self.npu_output = self.npu_model.output(0)
                print("NPU 模型加载成功！")
            else:
                self.npu_model = None
                print("NPU 模型文件不存在")
        else:
            self.npu_model = None
            print("NPU 不可用或检测模型不存在")

        # NPU 推理队列
        self.npu_queue = Queue(maxsize=2)
        self.npu_result = None
        self.npu_lock = threading.Lock()
        self.npu_running = True
        self.npu_infer_count = 0

        # 启动 NPU 线程
        if self.npu_model:
            self.npu_thread = threading.Thread(target=self._npu_worker, daemon=True)
            self.npu_thread.start()
            print("NPU 工作线程已启动")

        # 预热
        print("\n预热模型...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.pose_model(dummy, verbose=False)

        print("\n初始化完成！")
        print("=" * 60)

    def _npu_worker(self):
        """NPU 后台推理线程"""
        while self.npu_running:
            try:
                frame = self.npu_queue.get(timeout=0.1)
                if frame is None:
                    continue

                t_start = time.perf_counter()

                # 预处理
                h, w = frame.shape[:2]
                input_size = 640
                scale = input_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h))

                # 填充到 640x640
                padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
                padded[:new_h, :new_w] = resized

                # 转换为模型输入格式 [1, 3, 640, 640]
                blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
                blob = blob[np.newaxis, ...]

                t_preprocess = time.perf_counter()

                # NPU 推理
                result = self.npu_model([blob])[self.npu_output]

                t_infer = time.perf_counter()

                # 解析结果（简化：只统计检测数量）
                detections = result[0].T  # [8400, 84]
                confidences = detections[:, 4:].max(axis=1)
                count = np.sum(confidences > 0.5)

                t_end = time.perf_counter()

                with self.npu_lock:
                    self.npu_result = {
                        'count': int(count),
                        'preprocess_ms': (t_preprocess - t_start) * 1000,
                        'infer_ms': (t_infer - t_preprocess) * 1000,
                        'total_ms': (t_end - t_start) * 1000
                    }
                    self.npu_infer_count += 1

            except Empty:
                continue
            except Exception as e:
                pass

    def infer_pose(self, frame, conf=0.5, resize=480):
        """GPU 姿态估计"""
        h, w = frame.shape[:2]
        scale = resize / max(h, w)
        infer_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        results = self.pose_model(infer_frame, conf=conf, verbose=False)
        return results, (w / infer_frame.shape[1], h / infer_frame.shape[0])

    def submit_npu(self, frame):
        """提交帧到 NPU 处理"""
        if self.npu_model and not self.npu_queue.full():
            try:
                self.npu_queue.put_nowait(frame.copy())
            except:
                pass

    def get_npu_result(self):
        """获取 NPU 结果"""
        with self.npu_lock:
            return self.npu_result

    def stop(self):
        self.npu_running = False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='screen', choices=['camera', 'screen'])
    parser.add_argument('--monitor', type=int, default=0)
    parser.add_argument('--resize', type=int, default=480)
    parser.add_argument('--conf', type=float, default=0.5)
    args = parser.parse_args()

    # 初始化
    dual = DualInference()

    # 捕获源
    cap = None
    screen_cap = None

    if args.mode == 'camera':
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        screen_cap = ScreenCapture(args.monitor)

    window_name = 'GPU+NPU Pose [C=Cam, 1-9=Screen, ESC=Quit]'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n按键说明:")
    print("  C - 摄像头")
    print("  1-9 - 屏幕")
    print("  ESC - 退出")
    print()

    frame_count = 0
    start_time = time.time()
    fps = 0
    gpu_times = []
    last_print_time = time.time()
    last_npu_count = 0

    running = True
    mode = args.mode
    monitor_idx = args.monitor

    print("\n实时性能监控 (每2秒输出):")
    print("-" * 70)

    try:
        while running:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                break

            # 捕获
            if mode == 'camera' and cap:
                ret, frame = cap.read()
                if not ret:
                    continue
            else:
                frame = screen_cap.grab() if screen_cap else None
                if frame is None:
                    time.sleep(0.001)
                    continue

            frame_count += 1

            # 提交到 NPU（异步）
            if frame_count % 3 == 0:  # 每3帧提交一次
                dual.submit_npu(frame)

            # GPU 推理（同步）
            t0 = time.perf_counter()
            results, scale = dual.infer_pose(frame, args.conf, args.resize)
            gpu_time = (time.perf_counter() - t0) * 1000

            gpu_times.append(gpu_time)
            if len(gpu_times) > 30:
                gpu_times.pop(0)

            # 绘制结果
            display = frame.copy()
            scale_x, scale_y = scale

            # 骨架
            skeleton = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                       (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

            person_count = 0
            if results[0].keypoints is not None:
                person_count = len(results[0].keypoints)
                for kp in results[0].keypoints:
                    kp_xy = kp.xy[0].cpu().numpy()
                    kp_conf = kp.conf[0].cpu().numpy() if kp.conf is not None else np.ones(17)

                    kp_xy[:, 0] *= scale_x
                    kp_xy[:, 1] *= scale_y

                    for i, j in skeleton:
                        if kp_conf[i] > 0.5 and kp_conf[j] > 0.5:
                            pt1 = (int(kp_xy[i][0]), int(kp_xy[i][1]))
                            pt2 = (int(kp_xy[j][0]), int(kp_xy[j][1]))
                            cv2.line(display, pt1, pt2, (0, 255, 0), 2)

                    for idx, (x, y) in enumerate(kp_xy):
                        if kp_conf[idx] > 0.5:
                            cv2.circle(display, (int(x), int(y)), 5, (0, 0, 255), -1)

            # FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            avg_gpu = sum(gpu_times) / len(gpu_times) if gpu_times else 0

            # NPU 结果
            npu_result = dual.get_npu_result()
            npu_text = f"NPU: {npu_result['count']} obj" if npu_result else "NPU: --"

            # 显示信息
            h_disp, w_disp = display.shape[:2]
            texts = [
                (f"FPS: {fps:.1f} | People: {person_count}", (0, 255, 0)),
                (f"GPU: {avg_gpu:.1f}ms | {npu_text}", (0, 255, 255)),
                (f"Mode: {'Cam' if mode=='camera' else f'Screen{monitor_idx}'}", (200, 200, 200))
            ]

            for i, (text, color) in enumerate(reversed(texts)):
                sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.putText(display, text, (w_disp - sz[0] - 10, h_disp - 10 - i * 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow(window_name, display)

            # 每2秒输出性能统计
            current_time = time.time()
            if current_time - last_print_time >= 2.0:
                npu_result = dual.get_npu_result()
                npu_infer_count = dual.npu_infer_count
                npu_rate = (npu_infer_count - last_npu_count) / 2.0

                gpu_avg = sum(gpu_times) / len(gpu_times) if gpu_times else 0

                print(f"GPU: {gpu_avg:5.1f}ms ({1000/gpu_avg:4.1f} fps) | ", end="")
                print(f"NPU: ", end="")
                if npu_result and 'infer_ms' in npu_result:
                    print(f"{npu_result['infer_ms']:5.1f}ms ({npu_rate:.1f}/s) | ", end="")
                else:
                    print(f"  --  | ", end="")
                print(f"Total FPS: {fps:5.1f} | People: {person_count}")

                last_print_time = current_time
                last_npu_count = npu_infer_count

            # 按键
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('c') or key == ord('C'):
                if screen_cap:
                    screen_cap.release()
                    screen_cap = None
                if cap is None:
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                mode = 'camera'
            elif ord('1') <= key <= ord('9'):
                new_mon = key - ord('1')
                if cap:
                    cap.release()
                    cap = None
                if screen_cap:
                    screen_cap.release()
                monitor_idx = new_mon
                screen_cap = ScreenCapture(monitor_idx)
                mode = 'screen'

    finally:
        dual.stop()
        if cap:
            cap.release()
        if screen_cap:
            screen_cap.release()
        cv2.destroyAllWindows()

        # 最终统计
        total_time = time.time() - start_time
        gpu_avg = sum(gpu_times) / len(gpu_times) if gpu_times else 0

        print("\n" + "=" * 70)
        print("最终统计报告:")
        print("=" * 70)
        print(f"  运行时间: {total_time:.1f} 秒")
        print(f"  总帧数: {frame_count}")
        print(f"  平均 FPS: {frame_count/total_time:.1f}")
        print()
        print(f"  GPU 姿态估计:")
        print(f"    - 推理次数: {frame_count}")
        print(f"    - 平均耗时: {gpu_avg:.1f} ms")
        print(f"    - 理论 FPS: {1000/gpu_avg:.1f}")
        print()
        print(f"  NPU 目标检测:")
        print(f"    - 推理次数: {dual.npu_infer_count}")
        print(f"    - 推理频率: {dual.npu_infer_count/total_time:.1f} /秒")
        npu_result = dual.get_npu_result()
        if npu_result and 'infer_ms' in npu_result:
            print(f"    - 最后耗时: {npu_result['infer_ms']:.1f} ms")
        print()
        print("  并行效率分析:")
        print(f"    - GPU 利用率: ~{min(100, gpu_avg * fps / 10):.0f}%")
        if dual.npu_infer_count > 0:
            print(f"    - NPU 异步工作: ✓ (不阻塞主线程)")
        print("=" * 70)


if __name__ == "__main__":
    main()
