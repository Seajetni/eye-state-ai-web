# 🚀 คู่มือการรันและ Deploy Eye State AI ขึ้น Vercel

เว็บแอปพลิเคชันตรวจจับสถานะดวงตา (**ลืมตา / หลับตา**) และแจ้งเตือนอาการง่วง/หลับใน แบบ Real-time ผ่านกล้องเว็บแคม โดยรันโมเดล Deep Learning **`eye_cnn.onnx`** ใน Web Browser โดยตรงผ่าน **ONNX Runtime Web (WASM)** และ **MediaPipe Face Mesh**

---

## ⚡ วิธีทดสอบรันบนเครื่อง Local ทันที

เนื่องจากเป็น Client-Side Web Application คุณสามารถรันเว็บเซิร์ฟเวอร์ด้วย Python ที่มีอยู่ในเครื่องได้ทันที:

### 1. เปิด Terminal ในโฟลเดอร์ `D:\AI_CNN`
```powershell
python -m http.server 3000
```

### 2. เปิด Browser
เข้าไปที่:
👉 **[http://localhost:3000](http://localhost:3000)**

* กดปุ่ม **"เปิดกล้องเว็บแคม"**
* ให้สิทธิ์เข้าถึงกล้อง (Allow Camera)
* มองตรงไปยังกล้อง จะเห็นกรอบดวงตาซ้าย-ขวา พร้อมสถานะ **OPEN / CLOSED** และความมั่นใจแบบ Real-time (30-60 FPS)
* หากหลับตาทั้ง 2 ข้างต่อเนื่องเกิน 1.5 วินาที ระบบจะส่งเสียงแจ้งเตือน Beep Alert และแสดงแถบเตือนสีแดง!

---

## 🌐 ขั้นตอนการ Deploy ขึ้น Vercel (ฟรี 100%)

คุณสามารถนำขึ้น Vercel ได้ 2 วิธีหลัก:

### วิธีที่ 1: Deploy ผ่าน GitHub (แนะนำที่สุด - อัปเดตอัตโนมัติ)

1. **สร้าง Repository บน GitHub:**
   - เข้า [github.com](https://github.com) และสร้าง Repository ใหม่ เช่น `eye-state-ai-web`
2. **Push โค้ดขึ้น GitHub:**
   ```powershell
   git init
   git add index.html app.js vercel.json eye_cnn.onnx README.md
   git commit -m "Deploy Eye State AI to Vercel"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/eye-state-ai-web.git
   git push -u origin main
   ```
   *(หมายเหตุ: ไม่ต้อง push โฟลเดอร์ `dataset/` ขึ้น GitHub เพื่อประหยัดพื้นที่)*
3. **เชื่อมต่อกับ Vercel:**
   - เข้าสู่ระบบที่ [vercel.com](https://vercel.com)
   - กดปุ่ม **"Add New..."** ➔ **"Project"**
   - เลือก GitHub Repository `eye-state-ai-web`
   - Framework Preset: เลือก **"Other"** (Vercel จะตรวจจับไฟล์ `index.html` อัตโนมัติ)
   - กดปุ่ม **"Deploy"**
   - ภายในไม่กี่วินาที คุณจะได้ URL ใช้งานทันที เช่น: `https://eye-state-ai.vercel.app` 🎉

---

### วิธีที่ 2: Deploy ผ่าน Vercel CLI

หากต้องการรันคำสั่ง deploy จากเครื่องโดยตรง:

1. ติดตั้ง Vercel CLI (หากมี Node.js):
   ```powershell
   npm install -g vercel
   ```
2. รันคำสั่ง deploy:
   ```powershell
   vercel
   ```
   และเมื่อต้องการขึ้น Production:
   ```powershell
   vercel --prod
   ```

---

## 🛡️ ไฟล์คอนฟิก Vercel (`vercel.json`)

ในโปรเจกต์ได้จัดเตรียมไฟล์ `vercel.json` ซึ่งตั้งค่าสำคัญไว้แล้ว:
- **WASM Acceleration Headers:** เปิดใช้งาน `Cross-Origin-Opener-Policy` และ `Cross-Origin-Embedder-Policy` สำหรับ SharedArrayBuffer และ SIMD ใน WebAssembly
- **Model Caching:** แคชไฟล์ `eye_cnn.onnx` ไว้บน Vercel Global CDN เพื่อให้ผู้ใช้งานโหลดโมเดลได้อย่างรวดเร็วในเสี้ยววินาที
