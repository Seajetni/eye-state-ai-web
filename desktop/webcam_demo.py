import argparse
import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["Closed", "Open"]

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# Solves lighting glare, camera haze, and washed-out webcam exposure
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

def preprocess_eye_crop(crop_bgr, target_size=(64, 64)):
    """Preprocess eye crop for Kaggle EyeCNN model (64x64)."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def normalize_camera_source(src):
    if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
        return int(src)
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        src = src.rstrip("/")
        if not src.endswith(("/video", "/shot.jpg", ".mjpg", ".mp4")):
            src += "/video"
    return src

def get_crop_around_point(frame, cx, cy, size):
    half = size // 2
    h, w = frame.shape[:2]
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    if x2 <= x1 or y2 <= y1:
        return None, (0, 0, 0, 0)
    return frame[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

def main():
    parser = argparse.ArgumentParser(description="Real-time Eye State Detection (Webcam / IP Camera)")
    parser.add_argument("source", nargs="?", default="http://192.168.1.37:4747/video",
                        help="Camera source: index (e.g. 0) or IP Camera URL (default: http://192.168.1.37:4747/video)")
    parser.add_argument("--source", "-s", dest="source_opt", default=None,
                        help="Alternative flag for camera source")
    parser.add_argument("--model", "-m", dest="model_file", default="eye_cnn.onnx",
                        help="ONNX model file (default: eye_cnn.onnx)")
    parser.add_argument("--rotate", "-r", type=int, default=90, choices=[0, 90, 180, 270],
                        help="Rotate camera image in degrees (0, 90, 180, 270). Default: 90 (Portrait for DroidCam)")
    args = parser.parse_args()

    rotation_angle = args.rotate
    raw_source = args.source_opt if args.source_opt is not None else args.source
    camera_source = normalize_camera_source(raw_source)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")

    # Locate Eye CNN ONNX model
    if not os.path.isabs(args.model_file):
        model_path = os.path.join(models_dir, args.model_file)
        if not os.path.exists(model_path):
            model_path = os.path.join(script_dir, args.model_file)
    else:
        model_path = args.model_file

    print(f"[Info] Loading Eye CNN ONNX model from {model_path} ...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Check for YuNet DNN face and landmark detector
    yunet_path = os.path.join(models_dir, "yunet.onnx")
    if not os.path.exists(yunet_path):
        yunet_path = os.path.join(script_dir, "yunet.onnx")
    use_yunet = hasattr(cv2, "FaceDetectorYN") and os.path.exists(yunet_path)

    detector_yunet = None
    face_cascade = None

    if use_yunet:
        print(f"[Info] Using YuNet Deep Learning Face & Landmark Detector: {yunet_path}")
        detector_yunet = cv2.FaceDetectorYN.create(
            yunet_path, "", (320, 320), score_threshold=0.5, nms_threshold=0.3, top_k=5000
        )
    else:
        print("[Info] YuNet not available, falling back to Haar Cascade face detector.")
        face_xml = os.path.join(models_dir, "haarcascade_frontalface_default.xml")
        if not os.path.exists(face_xml):
            face_xml = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(face_xml)
        if face_cascade.empty():
            raise RuntimeError(f"[Error] Failed to load face cascade from '{face_xml}'")

    print(f"[Info] Connecting to camera source: {camera_source} ...")
    cap = cv2.VideoCapture(camera_source)

    # Check for DroidCam busy status
    if not cap.isOpened() and isinstance(camera_source, str) and camera_source.startswith("http"):
        try:
            import urllib.request
            req = urllib.request.Request(camera_source, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                content = resp.read(2048).decode("utf-8", errors="ignore")
                if "droidcam_busy" in content or "busy" in content.lower():
                    print("\n" + "=" * 65)
                    print("[❌ ข้อผิดพลาด: DroidCam ติดสถานะ Busy (มีโปรแกรมอื่นเปิดอยู่)]")
                    print("สาเหตุ: ในหน้าจอของคุณมี Google Chrome หรือโปรแกรมอื่นเปิดดูกล้องนี้อยู่")
                    print("เนื่องจาก DroidCam อนุญาตให้เชื่อมต่อได้ครั้งละ 1 โปรแกรมเท่านั้น!")
                    print("วิธีแก้: กรุณา 'ปิดแท็บ Google Chrome' ที่ดูกล้องอยู่ แล้วรันคำสั่งใหม่อีกครั้งครับ")
                    print("=" * 65 + "\n")
                    return
        except Exception:
            pass

    if not cap.isOpened() and camera_source != 0:
        print(f"[Warning] Could not connect to '{camera_source}'. Falling back to default webcam (0)...")
        camera_source = 0
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"\n[Error] Could not open camera source: {camera_source}")
        if isinstance(camera_source, str) and ("4747" in camera_source or "droidcam" in camera_source.lower()):
            print("[Tip] สำหรับ DroidCam: กรุณาปิดแท็บ Google Chrome ที่เปิดดูกล้องอยู่ก่อน แล้วรันใหม่ครับ")
        return

    print("[Info] Camera stream started. Press 'q' to quit, 'r' to rotate 90°.")

    prev_time = time.time()
    closed_start_time = None
    detector_size_set = None
    smoothed_probs = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] No frame received, waiting or stream ended...")
            time.sleep(0.05)
            continue

        # Apply rotation (default 90° portrait for DroidCam, toggled via 'r' key)
        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Resize high-res stream to 960px width for fast inference
        h, w = frame.shape[:2]
        if w > 960:
            target_w = 960
            target_h = int(h * (960 / w))
            frame = cv2.resize(frame, (target_w, target_h))
            h, w = target_h, target_w

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        detected_eyes = []
        face_boxes = []

        if use_yunet:
            if detector_size_set != (w, h):
                detector_yunet.setInputSize((w, h))
                detector_size_set = (w, h)

            _, faces = detector_yunet.detect(frame)
            if faces is not None:
                for idx, face in enumerate(faces):
                    fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    face_boxes.append((fx, fy, fw, fh))

                    rx, ry = int(face[4]), int(face[5])
                    lx, ly = int(face[6]), int(face[7])

                    inter_ocular = np.hypot(lx - rx, ly - ry)
                    eye_size = int(max(inter_ocular * 0.45, 44))

                    crop_r, box_r = get_crop_around_point(frame, rx, ry, eye_size)
                    crop_l, box_l = get_crop_around_point(frame, lx, ly, eye_size)

                    if crop_r is not None and crop_r.size > 0:
                        detected_eyes.append((crop_r, box_r, f"face_{idx}_R"))
                    if crop_l is not None and crop_l.size > 0:
                        detected_eyes.append((crop_l, box_l, f"face_{idx}_L"))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
            for idx, (fx, fy, fw, fh) in enumerate(faces):
                face_boxes.append((fx, fy, fw, fh))
                eye_w = int(0.32 * fw)
                eye_h = int(0.28 * fh)
                eye_y = fy + int(0.24 * fh)

                r_x = fx + int(0.15 * fw)
                l_x = fx + int(0.55 * fw)

                crop_r, box_r = get_crop_around_point(frame, r_x + eye_w // 2, eye_y + eye_h // 2, max(eye_w, eye_h))
                crop_l, box_l = get_crop_around_point(frame, l_x + eye_w // 2, eye_y + eye_h // 2, max(eye_w, eye_h))

                if crop_r is not None and crop_r.size > 0:
                    detected_eyes.append((crop_r, box_r, f"face_{idx}_R"))
                if crop_l is not None and crop_l.size > 0:
                    detected_eyes.append((crop_l, box_l, f"face_{idx}_L"))

        # Draw Face Boxes
        for (fx, fy, fw, fh) in face_boxes:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 140, 0), 2)

        # Classify Eyes with ONNX model
        closed_count = 0
        total_eyes = len(detected_eyes)

        for (crop, (ex, ey, ew, eh), eye_key) in detected_eyes:
            inp = preprocess_eye_crop(crop)
            outs = session.run(None, {input_name: inp})
            raw_probs = softmax(outs[0])[0]

            # Exponential moving average smoothing: 70% current, 30% history
            if eye_key in smoothed_probs:
                probs = 0.70 * raw_probs + 0.30 * smoothed_probs[eye_key]
            else:
                probs = raw_probs
            smoothed_probs[eye_key] = probs

            pred_idx = int(np.argmax(probs))
            label = CLASSES[pred_idx]
            conf = probs[pred_idx] * 100.0

            if label == "Closed":
                closed_count += 1
                color = (0, 0, 255)  # Red for closed
            else:
                color = (0, 255, 0)  # Green for open

            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, 2)
            cv2.putText(frame, f"{label} {conf:.0f}%", (ex, max(18, ey - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Drowsiness / Eyes Closed alert (triggered when eyes remain closed >= 1.5s)
        if total_eyes > 0 and closed_count == total_eyes:
            if closed_start_time is None:
                closed_start_time = time.time()
            duration = time.time() - closed_start_time
            if duration >= 1.5:
                cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 255), -1)
                cv2.putText(frame, f"WARNING: EYES CLOSED ({duration:.1f}s)!", (20, h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            closed_start_time = None

        # Info Header
        detector_name = "YuNet DNN" if use_yunet else "Haar Cascade"
        cv2.putText(frame, f"FPS: {fps:.1f} | {detector_name} | EyeCNN (Kaggle) | Rot: {rotation_angle}° ('r' to turn)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow("Eye Open/Closed Detection (ONNX)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            rotation_angle = (rotation_angle + 90) % 360
            detector_size_set = None
            print(f"[Info] Rotated view to {rotation_angle}°")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
