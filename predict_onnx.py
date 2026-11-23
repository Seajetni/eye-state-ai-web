import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

CLASSES = ["closed", "open"]

def preprocess_image(image_path, target_size=(64, 64)):
    """Preprocess image for ONNX model inference."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    
    # Normalize to [0, 1] then standardize
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    
    # HWC -> CHW -> NCHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def predict(image_path, model_path="D:/AI_CNN/eye_cnn.onnx"):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    input_data = preprocess_image(image_path)
    
    t0 = time.time()
    outputs = session.run(None, {input_name: input_data})
    inference_time = (time.time() - t0) * 1000
    
    logits = outputs[0]
    probs = softmax(logits)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx] * 100.0
    
    print("="*50)
    print(f"Image:          {image_path}")
    print(f"Prediction:     {pred_class.upper()}")
    print(f"Confidence:     {confidence:.2f}%")
    print(f"  - Closed:     {probs[0]*100:.2f}%")
    print(f"  - Open:       {probs[1]*100:.2f}%")
    print(f"Inference Time: {inference_time:.2f} ms")
    print("="*50)
    return pred_class, confidence

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Test on validation sample if no argument provided
        val_samples = list(Path("D:/AI_CNN/dataset/val").rglob("*.jpg"))
        if val_samples:
            img_path = str(val_samples[0])
            print(f"[Info] No image provided. Using sample: {img_path}")
        else:
            print("Usage: python predict_onnx.py <path_to_eye_image.jpg>")
            sys.exit(1)
            
    predict(img_path)
