from fastapi import FastAPI, File, UploadFile, HTTPException
from datetime import datetime, timezone
import uuid
import joblib
import tensorflow as tf
import numpy as np
import io
import requests
from PIL import Image
import gradio as gr

app = FastAPI(title="🛡️ SSS: AI Vision Module")

# 1. تحميل الموديلات
clf = joblib.load('face_svm_model.pkl')
feature_extractor = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg'
)

# 2. الـ Contract Mapping
PERSON_MAP = {
    "Person1": {"code": "face_Person_01", "name": "Person 1"},
    "Person2": {"code": "face_Person_02", "name": "Person 2"},
    "Person3": {"code": "face_Person_03", "name": "Person 3"},
    "Person4": {"code": "face_Person_04", "name": "Person 4"}
}

# 3. بيانات الربط مع الـ ASP.NET Backend
BACKEND_URL = "http://threes-3s.runasp.net/sensors/motion"
API_KEY = "THIS_IS_A _SUPER_SECRET_KEY_FOR_SMART_HOME_PROJECT_2025"

def run_inference(image: Image.Image):
    """دالة المعالجة والاستدلال المشتركة بين Gradio و FastAPI"""
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    embedding = feature_extractor.predict(img_array, verbose=0)
    probs = clf.predict_proba(embedding)[0]
    idx = np.argmax(probs)
    conf = float(probs[idx])

    class_names = list(PERSON_MAP.keys())
    predicted_label = class_names[idx]

    if conf >= 0.85:
        person_info = PERSON_MAP[predicted_label]
        detected_code = person_info["code"]
        detected_name = person_info["name"]
    else:
        detected_code = f"face_unknown_{uuid.uuid4().hex[:8]}"
        detected_name = "Unknown Person"

    return detected_code, detected_name, conf

@app.post("/predict")
async def predict_and_report(
    file: UploadFile = File(...), 
    sensorId: str = "unknown_sensor", 
    cameraId: str = "unknown_camera"
):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        detected_code, detected_name, conf = run_inference(img)

        payload = {
            "sensorId": sensorId, 
            "motionDetected": True,
            "homeOwnerUsername": "user_admin",
            "detectedPersonCode": detected_code,
            "detectedPersonName": detected_name,
            "recognitionConfidence": float(round(conf, 2)),
            "triggeredAtUtc": datetime.now(timezone.utc).isoformat(),
            "sourceSystem": "AI-Vision-Module",
            "cameraId": cameraId
        }

        # إرسال التقرير للباك إند
        headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
        try:
            requests.post(BACKEND_URL, json=payload, headers=headers, timeout=5)
        except Exception as e:
            print(f"⚠️ Backend reporting failed: {e}")

        return payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. واجهة Gradio لاختبار الموديل بالمتصفح
def gradio_predict(img):
    if img is None:
        return {"error": "يرجى رفع صورة للفحص"}
    code, name, conf = run_inference(img)
    return {
        "Person Name": name,
        "Person Code": code,
        "Confidence": f"{round(conf * 100, 2)}%"
    }

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=gr.JSON(label="Prediction Result"),
    title="🛡️ SSS AI: Face Recognition Module",
    description="FastAPI endpoints are running at /predict and /docs"
)

app = gr.mount_gradio_app(app, demo, path="/")