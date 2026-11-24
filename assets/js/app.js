// Eye State AI - Client-Side Real-time Detection
// Powered by ONNX Runtime Web & MediaPipe Face Mesh
// Warm Organic Ceramic Theme

const CLASSES = ["Closed", "Open"];
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let ortSession = null;
let faceMesh = null;
let camera = null;
let isRunning = false;
let isFlipped = true;
let isMuted = false;

// Performance & Tracking State
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;
let closedStartTime = null;
const smoothedProbs = {};

// DOM Elements
const videoEl = document.getElementById('webcam');
const overlayCanvas = document.getElementById('overlayCanvas');
const overlayCtx = overlayCanvas.getContext('2d');
const hiddenCropCanvas = document.getElementById('hiddenCropCanvas');
const hiddenCropCtx = hiddenCropCanvas.getContext('2d', { willReadFrequently: true });

const cropRightCanvas = document.getElementById('cropRightCanvas');
const cropRightCtx = cropRightCanvas.getContext('2d');
const cropLeftCanvas = document.getElementById('cropLeftCanvas');
const cropLeftCtx = cropLeftCanvas.getContext('2d');

const standbyRightIcon = document.getElementById('standbyRightIcon');
const standbyLeftIcon = document.getElementById('standbyLeftIcon');

const modelStatusEl = document.getElementById('modelStatus');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

const placeholder = document.getElementById('placeholder');
const btnStartPrompt = document.getElementById('btnStartPrompt');
const btnToggleCamera = document.getElementById('btnToggleCamera');
const btnCameraText = document.getElementById('btnCameraText');
const btnFlip = document.getElementById('btnFlip');
const flipStatus = document.getElementById('flipStatus');
const btnMute = document.getElementById('btnMute');
const muteStatus = document.getElementById('muteStatus');
const imageInput = document.getElementById('imageInput');

const videoFps = document.getElementById('videoFps');
const fpsCounter = document.getElementById('fpsCounter');
const latencyCounter = document.getElementById('latencyCounter');

const stateLabel = document.getElementById('stateLabel');
const stateSub = document.getElementById('stateSub');
const videoStateText = document.getElementById('videoStateText');
const stateDot = document.getElementById('stateDot');
const confidenceVal = document.getElementById('confidenceVal');
const overallBadge = document.getElementById('overallBadge');

const labelRight = document.getElementById('labelRight');
const labelLeft = document.getElementById('labelLeft');
const probOpenBar = document.getElementById('probOpenBar');
const probOpenText = document.getElementById('probOpenText');
const probClosedBar = document.getElementById('probClosedBar');
const probClosedText = document.getElementById('probClosedText');

const drowsyAlert = document.getElementById('drowsyAlert');
const drowsyText = document.getElementById('drowsyText');
const videoContainer = document.getElementById('videoContainer');

// Web Audio API for Drowsiness Beep Alarm
let audioCtx = null;
function playBeepAlarm() {
  if (isMuted) return;
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(659.25, audioCtx.currentTime); // E5 note (warm, pleasant chime)
    gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.35);
  } catch (e) {
    console.warn("Audio alarm error:", e);
  }
}

// 1. Initialize ONNX Runtime Model
async function initONNX() {
  try {
    statusText.textContent = "กำลังโหลดโมเดล ONNX (1.4MB)...";
    statusDot.className = "status-dot loading";

    // Set ONNX Runtime Web WASM options
    ort.env.wasm.numThreads = 2;
    ort.env.wasm.simd = true;

    // Load eye_cnn.onnx
    const modelPath = './assets/models/eye_cnn.onnx';
    ortSession = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm']
    });

    statusText.innerHTML = 'กล้องพร้อมทำงาน · <span id="fpsHeader">28 fps</span>';
    statusDot.className = "status-dot";
    console.log("[Info] ONNX model loaded successfully.");
  } catch (error) {
    console.error("[Error] Failed to load ONNX model:", error);
    statusText.textContent = "โหลดโมเดลไม่สำเร็จ: ตรวจสอบไฟล์ eye_cnn.onnx";
    statusDot.className = "status-dot error";
  }
}

// 2. Initialize MediaPipe Face Mesh
function initFaceMesh() {
  faceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
  });

  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  faceMesh.onResults(onResults);
}

// 3. Preprocess 64x64 image into NCHW Float32Array
function preprocessImage(canvas) {
  const ctx = canvas.getContext('2d');
  const imgData = ctx.getImageData(0, 0, 64, 64);
  const data = imgData.data;

  // 1 x 3 x 64 x 64 = 12,288 float elements
  const floatData = new Float32Array(1 * 3 * 64 * 64);
  const planeSize = 64 * 64;

  for (let i = 0; i < planeSize; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    // Normalize with ImageNet Mean & Std
    floatData[i] = (r - MEAN[0]) / STD[0];
    floatData[planeSize + i] = (g - MEAN[1]) / STD[1];
    floatData[2 * planeSize + i] = (b - MEAN[2]) / STD[2];
  }

  return new ort.Tensor('float32', floatData, [1, 3, 64, 64]);
}

// Softmax calculation
function softmax(logits) {
  const maxLogit = Math.max(logits[0], logits[1]);
  const exp0 = Math.exp(logits[0] - maxLogit);
  const exp1 = Math.exp(logits[1] - maxLogit);
  const sumExp = exp0 + exp1;
  return [exp0 / sumExp, exp1 / sumExp];
}

// 4. Run ONNX Inference on a single 64x64 canvas
async function runInference(cropCanvas) {
  if (!ortSession) return null;
  const tensor = preprocessImage(cropCanvas);
  const feeds = {};
  feeds[ortSession.inputNames[0]] = tensor;

  const t0 = performance.now();
  const results = await ortSession.run(feeds);
  const latency = performance.now() - t0;

  const logits = results[ortSession.outputNames[0]].data;
  const probs = softmax(logits); // [prob_closed, prob_open]
  return { probs, latency };
}

// Helper: Crop square from source
function cropEye(source, cx, cy, size, targetCanvas) {
  const half = size / 2;
  const x = Math.max(0, cx - half);
  const y = Math.max(0, cy - half);
  const ctx = targetCanvas.getContext('2d');
  ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  ctx.drawImage(source, x, y, size, size, 0, 0, targetCanvas.width, targetCanvas.height);
  return { x, y, size };
}

// 5. MediaPipe Result Handler
async function onResults(results) {
  const now = performance.now();
  frameCount++;
  if (now - lastFrameTime >= 500) {
    fps = (frameCount * 1000) / (now - lastFrameTime);
    const fpsText = `${Math.round(fps)} fps`;
    if (videoFps) videoFps.textContent = fpsText;
    const headerFpsSpan = document.getElementById('fpsHeader');
    if (headerFpsSpan) headerFpsSpan.textContent = fpsText;
    if (fpsCounter) fpsCounter.textContent = fps.toFixed(1);
    frameCount = 0;
    lastFrameTime = now;
  }

  overlayCanvas.width = videoEl.videoWidth || 640;
  overlayCanvas.height = videoEl.videoHeight || 480;
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
    stateLabel.textContent = "ไม่พบใบหน้า";
    stateSub.textContent = "กรุณาหันหน้าเข้าหากล้อง";
    videoStateText.textContent = "ไม่พบใบหน้า";
    stateLabel.style.color = "var(--text-secondary)";
    drowsyAlert.classList.add("hidden");
    videoContainer.classList.remove("drowsy-active");
    closedStartTime = null;
    return;
  }

  const landmarks = results.multiFaceLandmarks[0];
  const w = overlayCanvas.width;
  const h = overlayCanvas.height;

  // Key eye landmarks:
  // Right eye: 33 (outer), 133 (inner)
  // Left eye: 362 (inner), 263 (outer)
  const rOuter = { x: landmarks[33].x * w, y: landmarks[33].y * h };
  const rInner = { x: landmarks[133].x * w, y: landmarks[133].y * h };
  const rx = (rOuter.x + rInner.x) / 2;
  const ry = (rOuter.y + rInner.y) / 2;

  const lInner = { x: landmarks[362].x * w, y: landmarks[362].y * h };
  const lOuter = { x: landmarks[263].x * w, y: landmarks[263].y * h };
  const lx = (lOuter.x + lInner.x) / 2;
  const ly = (lOuter.y + lInner.y) / 2;

  const interOcular = Math.hypot(lx - rx, ly - ry);
  const eyeCropSize = Math.max(interOcular * 0.48, 44);

  // Crop eyes to 64x64 canvases
  const boxR = cropEye(videoEl, rx, ry, eyeCropSize, cropRightCanvas);
  const boxL = cropEye(videoEl, lx, ly, eyeCropSize, cropLeftCanvas);

  // Run AI Inference
  const resR = await runInference(cropRightCanvas);
  const resL = await runInference(cropLeftCanvas);

  if (!resR || !resL) return;

  const avgLatency = (resR.latency + resL.latency) / 2;
  if (latencyCounter) latencyCounter.textContent = Math.round(avgLatency);

  // Exponential Moving Average Smoothing
  function smooth(key, raw) {
    if (smoothedProbs[key]) {
      smoothedProbs[key] = [
        0.70 * raw[0] + 0.30 * smoothedProbs[key][0],
        0.70 * raw[1] + 0.30 * smoothedProbs[key][1]
      ];
    } else {
      smoothedProbs[key] = raw;
    }
    return smoothedProbs[key];
  }

  const probsR = smooth("R", resR.probs);
  const probsL = smooth("L", resL.probs);

  const isClosedR = probsR[0] > probsR[1];
  const isClosedL = probsL[0] > probsL[1];

  const confR = (isClosedR ? probsR[0] : probsR[1]) * 100;
  const confL = (isClosedL ? probsL[0] : probsL[1]) * 100;

  labelRight.textContent = `${isClosedR ? 'หลับตา' : 'ปกติ'} ${confR.toFixed(0)}%`;
  labelRight.style.color = isClosedR ? 'var(--accent-warning)' : 'var(--accent-sage)';

  labelLeft.textContent = `${isClosedL ? 'หลับตา' : 'ปกติ'} ${confL.toFixed(0)}%`;
  labelLeft.style.color = isClosedL ? 'var(--accent-warning)' : 'var(--accent-sage)';

  // Draw Minimalist Reticles on Overlay Canvas
  function drawBox(box, isClosed, label, conf) {
    overlayCtx.lineWidth = 2.2;
    overlayCtx.strokeStyle = isClosed ? '#c86d56' : '#778a6c';

    const x = box.x;
    const y = box.y;
    const s = box.size;

    overlayCtx.beginPath();
    if (overlayCtx.roundRect) {
      overlayCtx.roundRect(x, y, s, s, 10);
    } else {
      overlayCtx.strokeRect(x, y, s, s);
    }
    overlayCtx.stroke();

    // Sleek Pill Tag
    const tagW = 86;
    const tagH = 20;
    const tagY = Math.max(4, y - 24);
    overlayCtx.fillStyle = isClosed ? 'rgba(200, 109, 86, 0.92)' : 'rgba(119, 138, 108, 0.92)';
    overlayCtx.beginPath();
    if (overlayCtx.roundRect) {
      overlayCtx.roundRect(x, tagY, tagW, tagH, 6);
    } else {
      overlayCtx.fillRect(x, tagY, tagW, tagH);
    }
    overlayCtx.fill();

    overlayCtx.font = '500 11px Prompt, sans-serif';
    overlayCtx.fillStyle = '#ffffff';
    overlayCtx.fillText(`${label} ${conf.toFixed(0)}%`, x + 7, tagY + 14);
  }

  drawBox(boxR, isClosedR, isClosedR ? 'หลับตา' : 'ลืมตา', confR);
  drawBox(boxL, isClosedL, isClosedL ? 'หลับตา' : 'ลืมตา', confL);

  // Overall calculations
  const avgClosedProb = (probsR[0] + probsL[0]) / 2;
  const avgOpenProb = (probsR[1] + probsL[1]) / 2;

  probClosedBar.style.width = `${(avgClosedProb * 100).toFixed(1)}%`;
  probClosedText.textContent = `${(avgClosedProb * 100).toFixed(1)}%`;

  probOpenBar.style.width = `${(avgOpenProb * 100).toFixed(1)}%`;
  probOpenText.textContent = `${(avgOpenProb * 100).toFixed(1)}%`;

  const bothClosed = isClosedR && isClosedL;

  if (bothClosed) {
    stateLabel.textContent = "หลับตา (เตือน!)";
    videoStateText.textContent = "หลับตา (เตือน!)";
    stateSub.textContent = "ตรวจพบดวงตาปิดทั้งสองข้าง";
    stateLabel.style.color = "var(--accent-warning)";
    confidenceVal.textContent = `${(avgClosedProb * 100).toFixed(0)}%`;
    confidenceVal.style.color = "var(--accent-warning)";
    videoContainer.classList.add("drowsy-active");

    // Drowsiness Timer Check
    if (closedStartTime === null) {
      closedStartTime = performance.now();
    }
    const durationSec = (performance.now() - closedStartTime) / 1000;

    if (durationSec >= 1.5) {
      drowsyAlert.classList.remove("hidden");
      drowsyText.textContent = `⚠️ เตือน: ตรวจพบหลับตาต่อเนื่อง! (${durationSec.toFixed(1)}s)`;
      playBeepAlarm();
    }
  } else {
    closedStartTime = null;
    drowsyAlert.classList.add("hidden");
    videoContainer.classList.remove("drowsy-active");

    stateLabel.textContent = "ลืมตา (ปกติ)";
    videoStateText.textContent = "ลืมตา (ปกติ)";
    stateSub.textContent = "ตรวจพบใบหน้า 1 คน · ดวงตาปกติ";
    stateLabel.style.color = "var(--text-primary)";
    confidenceVal.textContent = `${(avgOpenProb * 100).toFixed(0)}%`;
    confidenceVal.style.color = "var(--text-primary)";
  }
}

// 6. Camera Controls
async function startCamera() {
  if (isRunning) return;
  try {
    placeholder.classList.add("hidden");
    cropRightCanvas.classList.remove("hidden");
    cropLeftCanvas.classList.remove("hidden");
    standbyRightIcon.classList.add("hidden");
    standbyLeftIcon.classList.add("hidden");
    labelRight.classList.remove("hidden");
    labelLeft.classList.remove("hidden");

    btnCameraText.textContent = "กำลังเปิดกล้อง...";

    camera = new Camera(videoEl, {
      onFrame: async () => {
        if (isRunning && faceMesh) {
          await faceMesh.send({ image: videoEl });
        }
      },
      width: 640,
      height: 480
    });

    await camera.start();
    isRunning = true;
    btnCameraText.textContent = "ปิดกล้อง";
    btnToggleCamera.classList.add("camera-active");
    console.log("[Info] Camera started successfully.");
  } catch (err) {
    console.error("[Error] Failed to access webcam:", err);
    alert("ไม่สามารถเข้าถึงกล้องเว็บแคมได้: กรุณาอนุญาตสิทธิ์กล้องในเบราว์เซอร์");
    stopCamera();
  }
}

function stopCamera() {
  if (camera) {
    camera.stop();
  }
  isRunning = false;
  placeholder.classList.remove("hidden");
  cropRightCanvas.classList.add("hidden");
  cropLeftCanvas.classList.add("hidden");
  standbyRightIcon.classList.remove("hidden");
  standbyLeftIcon.classList.remove("hidden");
  labelRight.classList.add("hidden");
  labelLeft.classList.add("hidden");

  btnCameraText.textContent = "สลับสถานะ";
  btnToggleCamera.classList.remove("camera-active");
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  drowsyAlert.classList.add("hidden");
  videoContainer.classList.remove("drowsy-active");

  stateLabel.textContent = "ลืมตา (ปกติ)";
  videoStateText.textContent = "ลืมตา (ปกติ)";
  stateSub.textContent = "ตรวจพบใบหน้า 1 คน · ดวงตาปกติ";
  stateLabel.style.color = "var(--text-primary)";
  confidenceVal.textContent = "96%";
  confidenceVal.style.color = "var(--text-primary)";
  probOpenBar.style.width = "93.4%";
  probOpenText.textContent = "93.4%";
  probClosedBar.style.width = "6.6%";
  probClosedText.textContent = "6.6%";
  if (videoFps) videoFps.textContent = "28 fps";
  const headerFpsSpan = document.getElementById('fpsHeader');
  if (headerFpsSpan) headerFpsSpan.textContent = "28 fps";
}

// 7. Event Listeners
btnToggleCamera.addEventListener('click', () => {
  if (isRunning) stopCamera();
  else startCamera();
});

btnStartPrompt.addEventListener('click', startCamera);

btnFlip.addEventListener('click', () => {
  isFlipped = !isFlipped;
  if (isFlipped) {
    videoEl.classList.remove('no-flip');
    overlayCanvas.classList.remove('no-flip');
    flipStatus.textContent = "เปิด";
  } else {
    videoEl.classList.add('no-flip');
    overlayCanvas.classList.add('no-flip');
    flipStatus.textContent = "ปิด";
  }
});

btnMute.addEventListener('click', () => {
  isMuted = !isMuted;
  muteStatus.textContent = isMuted ? "ปิด" : "เปิด";
});

// Image Upload Tester
imageInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  stopCamera();
  placeholder.classList.add("hidden");
  cropRightCanvas.classList.remove("hidden");
  cropLeftCanvas.classList.remove("hidden");
  standbyRightIcon.classList.add("hidden");
  standbyLeftIcon.classList.add("hidden");
  labelRight.classList.remove("hidden");
  labelLeft.classList.remove("hidden");

  const img = new Image();
  img.src = URL.createObjectURL(file);
  img.onload = async () => {
    overlayCanvas.width = img.width;
    overlayCanvas.height = img.height;
    overlayCtx.drawImage(img, 0, 0);
    if (faceMesh) {
      await faceMesh.send({ image: img });
    }
  };
});

// Startup Initializations
window.addEventListener('DOMContentLoaded', async () => {
  initFaceMesh();
  await initONNX();
});
