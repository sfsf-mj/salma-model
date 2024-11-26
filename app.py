from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "end3.keras"
IMG_SIZE = 256

# تحقق من وجود النموذج أثناء بدء التشغيل
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError("النموذج غير موجود! تأكد من رفع الملف قبل تشغيل التطبيق.")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")


try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# قائمة الفئات
categories = ['Apple___healthy', 'Apple___Black_rot']
# class_labels = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"]

def preprocess_image(image):
    try:
        image = image.resize((IMG_SIZE, IMG_SIZE))  # تغيير حجم الصورة
        image_array = np.array(image) / 255.0  # تطبيع القيم
        if image_array.shape[-1] != 3:
            raise ValueError("الصورة ليست بتنسيق RGB")
        return image_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)  # إعادة تشكيل البيانات
    except Exception as e:
        raise ValueError("خطأ في معالجة الصورة: " + str(e))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم إرسال ملف الصورة. تأكد من إرسال صورة بصيغة مدعومة (PNG أو JPG أو JPEG)."}), 400
    
    file = request.files['file']
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "نوع الملف غير مدعوم. يرجى تحميل صورة."}), 400

    try:
        image = Image.open(file.stream)
        image.verify()  # التحقق من أن الملف هو صورة حقيقية
        image = Image.open(file.stream)  # إعادة فتح الصورة إذا كانت صالحة
    except Exception:
        return jsonify({"error": "الملف ليس صورة صالحة"}), 400

    try:
        preprocessed_image = preprocess_image(image)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # إجراء التنبؤ
    prediction = model.predict(preprocessed_image)
    predicted_class = int(prediction[0][0] > 0.5)  # تحويل الاحتمالية إلى 0 أو 1
    confidence = float(prediction[0][0]) if predicted_class == 1 else float(1 - prediction[0][0])

    # إعداد الاستجابة
    response = {
        "predicted_label": categories[predicted_class],
        "confidence": confidence
    }

    return jsonify(response)
    # predictions = model.predict(preprocessed_image)
    # class_index = np.argmax(predictions, axis=1)[0]
    # confidence = np.max(predictions)

    # response = {
    #     "class_index": int(class_index),
    #     "class_label": class_labels[class_index],
    #     "confidence": float(confidence)
    # }
    # return jsonify(response)

if __name__ == '__main__':
    app.run()

# if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
