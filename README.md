# AI Eye State Classification (Open vs Closed Eyes)

โปรเจกต์ Deep Learning สำหรับตรวจจับและจำแนกสถานะดวงตา (**ลืมตา / หลับตา**) แบบ Real-time ผ่านกล้อง Webcam / IP Camera (DroidCam) จากชุดข้อมูล **Kaggle Dataset** และ Export เป็นโมเดล **ONNX** เพื่อประสิทธิภาพความเร็วสูงสุด

---

## 📁 โครงสร้างโปรเจกต์ (`D:\AI_CNN`)

```text
D:\AI_CNN\
├── dataset\                   # ชุดข้อมูลภาพดวงตาจาก Kaggle (แบ่ง Train 80% / Val 20%)
│   ├── train\ (1,160 ภาพ)     # closed: 580 ภาพ, open: 580 ภาพ
│   └── val\   (292 ภาพ)       # closed: 146 ภาพ, open: 146 ภาพ
├── model.py                   # โครงสร้างโมเดล EyeCNN (3 Conv Blocks + Classifier)
├── prepare_data.py            # สคริปต์ดาวน์โหลดข้อมูลจาก Kaggle และจัดหมวดหมู่
├── train.py                   # สคริปต์เทรนโมเดล 10 Epochs และ Export เป็น ONNX
├── predict_onnx.py            # สคริปต์ทดสอบ Inference รูปภาพด้วย ONNX Runtime
├── webcam_demo.py             # เดโมตรวจจับดวงตา Real-time ผ่านกล้องเว็บแคม / DroidCam
├── yunet.onnx                 # โมเดล Deep Learning Face & Landmark Detector (OpenCV YuNet)
├── haarcascade_eye.xml        # ไฟล์ Haar Cascade สำหรับตรวจจับดวงตา (Fallback)
├── haarcascade_frontalface_default.xml # ไฟล์ Haar Cascade สำหรับตรวจจับใบหน้า (Fallback)
├── best_model.pth             # โมเดล PyTorch Weights ที่มีความแม่นยำสูงสุด
├── eye_cnn.onnx               # โมเดล ONNX สำหรับ Production / Deployment
└── requirements.txt           # รายการไลบรารีที่จำเป็น
```

---

## 📊 ข้อมูล Dataset และผลการเทรน (Kaggle Dataset)

* **ที่มา Dataset:** Kaggle (`dheerajperumandla/drowsiness-dataset`)
* **จำนวนรูปทั้งหมด:** 1,452 รูป (Open: 726 รูป, Closed: 726 รูป)
* **สัดส่วนการแบ่ง:**
  * Training Set: 1,160 รูป (Balanced 50/50)
  * Validation Set: 292 รูป (Balanced 50/50)
* **ความแม่นยำสูงสุด (Validation Accuracy):** **98.97%**
* **Inference Time:** < 1 มิลลิวินาที (ms) ต่อภาพบน CPU

---

## 🚀 วิธีการรันใช้งาน

### 1. รันตรวจจับแบบ Real-time ผ่านกล้อง (`webcam_demo.py`)

- **เชื่อมต่อกับ IP Webcam / DroidCam:**
```powershell
& 'C:\Users\🧙‍♂️\AppData\Local\Programs\Python\Python311\python.exe' 'd:/AI_CNN/webcam_demo.py'
```
*(ค่าเริ่มต้นจะเชื่อมต่อไปยัง `http://192.168.1.37:4747/video` และหมุนภาพเป็นแนวตั้ง 90° ให้อัตโนมัติ)*

* **คีย์ลัดระหว่างเปิดกล้อง:**
  * กดปุ่ม **`r`** บนคีย์บอร์ด เพื่อหมุนภาพทีละ 90° (0° ➔ 90° ➔ 180° ➔ 270°)
  * กดปุ่ม **`q`** บนคีย์บอร์ด เพื่อปิดโปรแกรม

- **หรือใช้กล้องเว็บแคมของคอมพิวเตอร์ (Webcam 0):**
```powershell
& 'C:\Users\🧙‍♂️\AppData\Local\Programs\Python\Python311\python.exe' 'd:/AI_CNN/webcam_demo.py' 0
```

---

### 2. ทดสอบ Inference รูปภาพเดี่ยว (`predict_onnx.py`)

```powershell
& 'C:\Users\🧙‍♂️\AppData\Local\Programs\Python\Python311\python.exe' 'd:/AI_CNN/predict_onnx.py'
```
หรือระบุไฟล์ภาพดวงตาที่ต้องการทดสอบ:
```powershell
& 'C:\Users\🧙‍♂️\AppData\Local\Programs\Python\Python311\python.exe' 'd:/AI_CNN/predict_onnx.py' 'D:/AI_CNN/dataset/val/open/_107.jpg'
```

---

### 3. รันเทรนใหม่หรือเทรนซ้ำ (`train.py`)

```powershell
& 'C:\Users\🧙‍♂️\AppData\Local\Programs\Python\Python311\python.exe' 'd:/AI_CNN/train.py' --epochs 10
```
*(ระบบจะเทรน 10 Epochs บน Kaggle Dataset, เซฟ `best_model.pth` และ Export เป็น `eye_cnn.onnx` ให้อัตโนมัติ)*
