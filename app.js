// Eye State AI - Client-Side Real-time Detection
// Powered by ONNX Runtime Web & MediaPipe Face Mesh

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

const modelStatusEl = document.getElementById('modelStatus');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

const placeholder = document.getElementById('placeholder');
const btnStartPrompt = document.getElementById('btnStartPrompt');
const btnToggleCamera = document.getElementById('btnToggleCamera');
const btnCameraText = document.getElementById('btnCameraText');
const btnFlip = document.getElementById('btnFlip');
const btnMute = document.getElementById('btnMute');
const muteStatus = document.getElementById('muteStatus');
const imageInput = document.getElementById('imageInput');

const fpsCounter = document.getElementById('fpsCounter');
const latencyCounter = document.getElementById('latencyCounter');
const stateLabel = document.getElementById('stateLabel');
const stateSub = document.getElementById('stateSub');
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
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime); // 880 Hz (A5)
    gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.3);
  } catch (e) {
    console.warn("Audio alarm error:", e);
  }
}

// 1. Initialize ONNX Runtime Model
async function initONNX() {
  try {
    statusText.textContent = "กำลังโหลดโมเดล ONNX (1.4MB)...";
    statusDot.className = "w-2 h-2 rounded-full bg-amber-400 animate-pulse";

    // Set ONNX Runtime Web WASM options
    ort.env.wasm.numThreads = 2;
    ort.env.wasm.simd = true;

    // Load eye_cnn.onnx
    const modelPath = './eye_cnn.onnx';
    ortSession = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm']
    });

    statusText.textContent = "โมเดล AI พร้อมทำงาน";
    statusDot.className = "w-2 h-2 rounded-full bg-emerald-400";
    modelStatusEl.classList.add("border-emerald-500/40");
    console.log("[Info] ONNX model loaded successfully.");
  } catch (error) {
    console.error("[Error] Failed to load ONNX model:", error);
    statusText.textContent = "โหลดโมเดลไม่สำเร็จ: ตรวจสอบไฟล์ eye_cnn.onnx";
    statusDot.className = "w-2 h-2 rounded-full bg-rose-500";
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
    fpsCounter.textContent = fps.toFixed(1);
    frameCount = 0;
    lastFrameTime = now;
  }

  overlayCanvas.width = videoEl.videoWidth || 640;
  overlayCanvas.height = videoEl.videoHeight || 480;
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
    stateLabel.textContent = "ไม่พบใบหน้า";
    stateSub.textContent = "กรุณาหันหน้าเข้าหากล้อง";
    stateDot.className = "w-4 h-4 rounded-full bg-slate-600";
    drowsyAlert.classList.add("hidden");
    videoContainer.classList.remove("drowsy-active");
    closedStartTime = null;
    return;
  }

  const landmarks = results.multiFaceLandmarks[0];
  const w = overlayCanvas.width;
  const h = overlayCanvas.height;

  // Key eye landmarks:
  // Right eye (user's right): 33 (outer), 133 (inner)
  // Left eye (user's left): 362 (inner), 263 (outer)
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

  // Crop eyes
  const boxR = cropEye(videoEl, rx, ry, eyeCropSize, cropRightCanvas);
  const boxL = cropEye(videoEl, lx, ly, eyeCropSize, cropLeftCanvas);

  // Run AI Inference
  const resR = await runInference(cropRightCanvas);
  const resL = await runInference(cropLeftCanvas);

  if (!resR || !resL) return;

  const avgLatency = (resR.latency + resL.latency) / 2;
  latencyCounter.textContent = Math.round(avgLatency);

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

  labelRight.textContent = `${isClosedR ? 'Closed' : 'Open'} (${confR.toFixed(0)}%)`;
  labelRight.style.color = isClosedR ? '#fb7185' : '#34d399';

  labelLeft.textContent = `${isClosedL ? 'Closed' : 'Open'} (${confL.toFixed(0)}%)`;
  labelLeft.style.color = isClosedL ? '#fb7185' : '#34d399';

  // Draw Boxes on Overlay Canvas
  function drawBox(box, isClosed, label, conf) {
    overlayCtx.lineWidth = 2.5;
    overlayCtx.strokeStyle = isClosed ? '#f43f5e' : '#10b981';

    const x = box.x;
    const y = box.y;
    const s = box.size;

    overlayCtx.beginPath();
    if (overlayCtx.roundRect) {
      overlayCtx.roundRect(x, y, s, s, 8);
    } else {
      overlayCtx.strokeRect(x, y, s, s);
    }
    overlayCtx.stroke();

    // Sleek Tag Pill
    const tagW = 94;
    const tagH = 22;
    const tagY = Math.max(4, y - 26);
    overlayCtx.fillStyle = isClosed ? 'rgba(225, 29, 72, 0.9)' : 'rgba(16, 185, 129, 0.9)';
    overlayCtx.beginPath();
    if (overlayCtx.roundRect) {
      overlayCtx.roundRect(x, tagY, tagW, tagH, 6);
    } else {
      overlayCtx.fillRect(x, tagY, tagW, tagH);
    }
    overlayCtx.fill();

    overlayCtx.font = 'bold 11px monospace';
    overlayCtx.fillStyle = '#ffffff';
    overlayCtx.fillText(`${label.toUpperCase()} ${conf.toFixed(0)}%`, x + 8, tagY + 15);
  }

  drawBox(boxR, isClosedR, isClosedR ? 'Closed' : 'Open', confR);
  drawBox(boxL, isClosedL, isClosedL ? 'Closed' : 'Open', confL);

  // Overall calculations
  const avgClosedProb = (probsR[0] + probsL[0]) / 2;
  const avgOpenProb = (probsR[1] + probsL[1]) / 2;

  probClosedBar.style.width = `${(avgClosedProb * 100).toFixed(1)}%`;
  probClosedText.textContent = `${(avgClosedProb * 100).toFixed(1)}%`;

  probOpenBar.style.width = `${(avgOpenProb * 100).toFixed(1)}%`;
  probOpenText.textContent = `${(avgOpenProb * 100).toFixed(1)}%`;

  const bothClosed = isClosedR && isClosedL;

  if (bothClosed) {
    stateLabel.textContent = "หลับตา (CLOSED)";
    stateSub.textContent = "ตรวจพบดวงตาปิดทั้งสองข้าง";
    stateLabel.style.color = "#fb7185";
    stateDot.style.background = "#f43f5e";
    stateDot.style.boxShadow = "0 0 14px rgba(244, 63, 94, 0.8)";
    confidenceVal.textContent = `${(avgClosedProb * 100).toFixed(0)}%`;
    confidenceVal.style.color = "#fb7185";
    overallBadge.style.borderColor = "rgba(244, 63, 94, 0.4)";
    overallBadge.style.background = "rgba(225, 29, 72, 0.12)";

    // Drowsiness Timer Check
    if (closedStartTime === null) {
      closedStartTime = performance.now();
    }
    const durationSec = (performance.now() - closedStartTime) / 1000;

    if (durationSec >= 1.5) {
      drowsyAlert.classList.remove("hidden");
      drowsyText.textContent = `⚠️ เตือน: ตรวจพบหลับตาต่อเนื่อง! (${durationSec.toFixed(1)}s)`;
      videoContainer.classList.add("drowsy-active");
      playBeepAlarm();
    }
  } else {
    closedStartTime = null;
    drowsyAlert.classList.add("hidden");
    videoContainer.classList.remove("drowsy-active");

    stateLabel.textContent = "ลืมตา (OPEN)";
    stateSub.textContent = "ระบบตรวจจับสถานะดวงตาปกติ";
    stateLabel.style.color = "#34d399";
    stateDot.style.background = "#10b981";
    stateDot.style.boxShadow = "0 0 14px rgba(16, 185, 129, 0.8)";
    confidenceVal.textContent = `${(avgOpenProb * 100).toFixed(0)}%`;
    confidenceVal.style.color = "#34d399";
    overallBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
    overallBadge.style.background = "rgba(16, 185, 129, 0.12)";
  }
}

// 6. Camera Controls
async function startCamera() {
  if (isRunning) return;
  try {
    placeholder.classList.add("hidden");
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
    btnToggleCamera.classList.add("btn-danger");
    console.log("[Info] Camera started successfully.");
  } catch (err) {
    console.error("[Error] Failed to access webcam:", err);
    alert("ไม่สามารถเข้าถึงกล้องเว็บแคมได้: กรุณาอนุญาตสิทธิ์กล้องในเบราว์เซอร์");
    placeholder.classList.remove("hidden");
    btnCameraText.textContent = "เปิดกล้อง";
  }
}

function stopCamera() {
  if (!isRunning) return;
  if (camera) {
    camera.stop();
  }
  isRunning = false;
  placeholder.classList.remove("hidden");
  btnCameraText.textContent = "เปิดกล้อง";
  btnToggleCamera.classList.remove("btn-danger");
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  drowsyAlert.classList.add("hidden");
  videoContainer.classList.remove("drowsy-active");
  stateLabel.textContent = "กล้องปิดอยู่";
  stateLabel.style.color = "#94a3b8";
  stateDot.style.background = "#475569";
  stateDot.style.boxShadow = "none";
  overallBadge.style.borderColor = "rgba(255, 255, 255, 0.06)";
  overallBadge.style.background = "rgba(15, 23, 42, 0.75)";
}

// 7. Event Listeners
btnToggleCamera.addEventListener('click', () => {
  if (isRunning) stopCamera();
  else startCamera();
});

btnStartPrompt.addEventListener('click', startCamera);

btnFlip.addEventListener('click', () => {
  isFlipped = !isFlipped;
  const scaleClass = isFlipped ? '-scale-x-100' : 'scale-x-100';
  videoEl.className = `w-full h-full object-cover transform ${scaleClass}`;
  overlayCanvas.className = `absolute inset-0 w-full h-full object-cover transform ${scaleClass} pointer-events-none`;
});

btnMute.addEventListener('click', () => {
  isMuted = !isMuted;
  muteStatus.textContent = isMuted ? "ปิด" : "เปิด";
  btnMute.classList.toggle("text-rose-400", isMuted);
});

// Image Upload Tester
imageInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  stopCamera();
  placeholder.classList.add("hidden");

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
