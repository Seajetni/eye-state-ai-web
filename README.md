# 👁️ Eye State AI - Biometric Detection & Telemetry System

โปรเจกต์ Deep Learning สำหรับตรวจจับและจำแนกสถานะดวงตา (**ลืมตา / หลับตา**) และตรวจจับอาการง่วง/หลับในแบบ Real-time รองรับทั้ง **Web Application (Vercel)** และ **Desktop Application (OpenCV/PyTorch)**

---

## 📁 โครงสร้างโปรเจกต์ที่เป็นระเบียบ (Clean Architecture)

```text
D:\AI_CNN\
│
├── 🌐 assets/                           # ไฟล์และโมเดลสำหรับ Web Application
│   ├── css/
│   │   └── style.css                    # สไตล์ธีมโมเดิร์น Biometric Glassmorphism (Pure CSS)
│   ├── js/
│   │   └── app.js                       # WebRTC + MediaPipe Face Mesh + ONNX Runtime Web
│   └── models/
│       └── eye_cnn.onnx                 # โมเดล ONNX สำหรับ Web Inference (1.42 MB)
│
├── 🖥️ desktop/                          # โปรแกรม Desktop & สคริปต์ AI (Python)
│   ├── models/                          # โฟลเดอร์รวมโมเดลและ Cascade สำหรับ Desktop
│   │   ├── eye_cnn.onnx                 # โมเดล ONNX สำหรับ Production
│   │   ├── yunet.onnx                   # Face & Landmark Detector (OpenCV YuNet)
│   │   ├── best_model.pth               # PyTorch Model Weights
│   │   ├── haarcascade_eye.xml          # Haar Cascade ตรวจจับตา (Fallback)
│   │   └── haarcascade_frontalface_default.xml # Haar Cascade ตรวจจับหน้า (Fallback)
│   ├── model.py                         # โครงสร้างโมเดล PyTorch EyeCNN (3 Conv Blocks)
│   ├── prepare_data.py                  # สคริปต์ดาวน์โหลดและจัดสรร Kaggle Dataset
│   ├── train.py                         # สคริปต์เทรนโมเดล และ Export เป็น ONNX อัตโนมัติ
│   ├── predict_onnx.py                  # สคริปต์ทดสอบ Inference รูปเดี่ยว
│   ├── webcam_demo.py                   # โปรแกรมเปิดกล้อง Real-time บนคอมพิวเตอร์
│   └── requirements.txt                 # ไลบรารี Python สำหรับ Desktop
│
├── 📊 dataset/                          # ชุดข้อมูลภาพดวงตาจาก Kaggle (Train 80% / Val 20%)
│   ├── train/ (1,160 ภาพ)
│   └── val/   (292 ภาพ)
│
├── index.html                           # หน้า Web UI หลักสำหรับ Production
├── vercel.json                          # คอนฟิก Vercel Static Hosting
├── .vercelignore                        # กรองไฟล์ Desktop/Dataset ออกจาก Vercel
├── .gitignore                           # กรองไฟล์ขนาดใหญ่ไม่ให้ขึ้น Git
└── README.md                            # คู่มือการใช้งานระบบ
```

---

## 🌐 1. การใช้งานบน Web Application (Vercel)

* **Production URL:** [https://eye-state-ai-web-fe9e.vercel.app](https://eye-state-ai-web-fe9e.vercel.app)
* **การทำงาน:** ประมวลผลแบบ Client-side 100% ในเบราว์เซอร์ผ่าน WebAssembly (WASM) และ MediaPipe Face Mesh โดยไม่ต้องพึ่งพาเซิร์ฟเวอร์ภายนอก

### ทดสอบ Web บนเครื่อง Local:
```powershell
python -m http.server 3000
```
เปิดเบราว์เซอร์ไปที่: `http://localhost:3000`

---

## 🖥️ 2. การใช้งานบน Desktop Application (Python)

### ติดตั้งไลบรารี:
```powershell
cd D:\AI_CNN\desktop
pip install -r requirements.txt
```

### เปิดโปรแกรมกล้อง Real-time บนเครื่อง:
```powershell
python desktop/webcam_demo.py
```
* **ใช้กล้องเว็บแคมในเครื่อง (Webcam 0):**
  ```powershell
  python desktop/webcam_demo.py 0
  ```
* **คีย์ลัดระหว่างเปิดกล้อง:**
  * กด **`r`** เพื่อหมุนภาพทีละ 90° (เหมาะสำหรับกล้องโทรศัพท์ DroidCam)
  * กด **`q`** เพื่อปิดโปรแกรม

---

## 🧠 3. การเทรนโมเดลใหม่ (Training & Export)

```powershell
python desktop/train.py --epochs 10
```
* สคริปต์จะเทรนโมเดลบน Kaggle Dataset บันทึกน้ำหนัก `best_model.pth` และ Export เป็น `eye_cnn.onnx` ไปยังทั้ง `desktop/models/` และ `assets/models/` ให้อัตโนมัติ

---

## 🖼️ 4. ทดสอบรูปภาพเดี่ยว (CLI Inference)

```powershell
python desktop/predict_onnx.py
```
หรือระบุไฟล์รูปภาพ:
```powershell
python desktop/predict_onnx.py dataset/val/open/_107.jpg
```
