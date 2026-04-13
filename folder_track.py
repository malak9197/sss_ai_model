import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
import numpy as np
import joblib
import tensorflow as tf


model_path = 'D:\\Malak\\sss_ai\\face_svm_model.pkl' 
clf = joblib.load(model_path)
feature_extractor = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg')
class_names = ['Person1', 'Person2', 'Person3', 'Person4']

class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
       
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"📸 New image detected: {os.path.basename(event.src_path)}")
            time.sleep(0.5) # وقت بسيط للتأكد إن الملف اتحفظ بالكامل
            self.process_image(event.src_path)

    def process_image(self, img_path):
        img = cv2.imread(img_path)
        if img is not None:
           
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_res = cv2.resize(img_rgb, (224, 224))
            img_array = np.expand_dims(img_res / 255.0, axis=0)
            
          
            embedding = feature_extractor.predict(img_array, verbose=0)
            probs = clf.predict_proba(embedding)
            idx = np.argmax(probs)
            conf = np.max(probs)
            
            name = class_names[idx] if conf > 0.85 else "Unknown"
            print(f"🎯 Result: {name} | Confidence: {conf*100:.2f}%")
            print("-" * 30)


path_to_watch = "D:\\Malak\\sss_ai\\images" # الفولدر اللي عايزة تراقيبه
if not os.path.exists(path_to_watch): os.makedirs(path_to_watch)

event_handler = ImageHandler()
observer = Observer()
observer.schedule(event_handler, path_to_watch, recursive=False)

print(f"👀 Monitoring folder: {path_to_watch} ... Press Ctrl+C to stop.")
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()