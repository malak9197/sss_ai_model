from fastapi import FastAPI, File, UploadFile, HTTPException
from datetime import datetime, timezone
import uuid
import joblib
import tensorflow as tf
import numpy as np
import io
import requests
from PIL import Image

app = FastAPI(title="🛡️ SSS: AI Vision Module")

# 1. تحميل الموديلات (تأكدي من وجود الملفات في نفس المسار)
clf = joblib.load('face_svm_model.pkl')
feature_extractor = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg'
)

# 2. الـ Contract Mapping (الأكواد الثابتة للأشخاص)
PERSON_MAP = {
    "Person1": {"code": "face_Person_01", "name": "Person 1"},
    "Person2": {"code": "face_Person_02", "name": "Person 2"},
    "Person3": {"code": "face_Person_03", "name": "Person 3"},
    "Person4": {"code": "face_Person_04", "name": "Person 4"}
}

# 3. بيانات الربط مع الـ ASP.NET Backend
BACKEND_URL = "http://smart3s.runasp.net/api/alerts/motion/report"
API_KEY = "CHANGE-THIS-TO-A-STRONG-SECRET-KEY"

@app.post("/predict")
async def predict_and_report(
    file: UploadFile = File(...), 
    sensorId: str = "unknown_sensor", 
    cameraId: str = "unknown_camera"
):
    try:
        # أ. معالجة الصورة (التحويل لـ 224x224)
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ب. استخراج الميزات (1280 Features) والتوقع بالـ SVM
        embedding = feature_extractor.predict(img_array, verbose=0)
        probs = clf.predict_proba(embedding)
        idx = np.argmax(probs)
        conf = np.max(probs)

        # ج. تحديد الهوية بناءً على الـ Confidence (85%)
        class_names = list(PERSON_MAP.keys())
        predicted_label = class_names[idx]
        
        if conf >= 0.85:
            person_info = PERSON_MAP[predicted_label]
            detected_code = person_info["code"]
            detected_name = person_info["name"]
        else:
            detected_code = f"face_unknown_{uuid.uuid4().hex[:8]}"
            detected_name = "Unknown Person"

        # د. تجهيز الـ JSON النهائي (الـ Contract)
        payload = {
            "sensorId": sensorId, # بيجي من الـ Request بناءً على الحساس اللي لقط الحركة
            "motionDetected": True,
            "homeOwnerUsername": "user_admin",
            "detectedPersonCode": detected_code,
            "detectedPersonName": detected_name,
            "recognitionConfidence": float(round(conf, 2)),
            "triggeredAtUtc": datetime.now(timezone.utc).isoformat(),
            "sourceSystem": "AI-Vision-Module",
            "cameraId": cameraId # بيجي من الـ Request بناءً على الكاميرا اللي صورت
        }

        # هـ. إرسال التقرير فوراً للباك إند بالـ API Key
        headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
        try:
            requests.post(BACKEND_URL, json=payload, headers=headers, timeout=5)
        except Exception as e:
            print(f"⚠️ Backend reporting failed: {e}")

        return payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))