from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import zipfile
import stat

app = Flask(__name__)

# تعريف مسار الملف
model_path = os.path.join(os.path.dirname(__file__), "end3.keras")

# التحقق من وجود الملف
if os.path.exists(model_path):
    print("ChatGPT ask: ✔ File exists:", model_path)

    # 1. التحقق مما إذا كان الملف تالفًا أو غير مكتمل
    try:
        # محاولة فتح الملف كملف مضغوط
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            print("ChatGPT ask: ✔ The file is a valid .keras zip file.")
            # عرض محتويات الملف للتحقق من البنية
            print("ChatGPT ask: Contents:", zip_ref.namelist())
    except zipfile.BadZipFile:
        print("ChatGPT ask: ✘ The file is corrupted or not a valid zip file.")
    except Exception as e:
        print("ChatGPT ask: ✘ An error occurred while checking the file:", str(e))
    
    # 2. التحقق من نوع الملف الصحيح
    try:
        # محاولة تحميل النموذج باستخدام TensorFlow
        model = tf.keras.models.load_model(model_path)
        print("ChatGPT ask: ✔ The file is a valid TensorFlow model.")
    except Exception as e:
        print("ChatGPT ask: ✘ The file is not a valid TensorFlow model:", str(e))
else:
    print("ChatGPT ask: ✘ File not found! Please check the file path.")

# 3. طباعة إصدار TensorFlow
print("ChatGPT ask: TensorFlow version:", tf.__version__)

# import os
print("ChatGPT say File exists:", os.path.exists("/app/end3.keras"))

# تحميل النموذج
# if not os.path.exists("end3.keras"):
#     return jsonify({"error": "النموذج غير موجود!"}), 500
# model = tf.keras.models.load_model("end3.keras")
    
def check_model_file(file_path):
    """
    التحقق مما إذا كان ملف النموذج .keras صالحًا أو تالفًا.
    
    Args:
        file_path (str): المسار إلى ملف النموذج.
    
    Returns:
        dict: يحتوي على حالة التحقق ورسالة.
    """
    try:
        # محاولة تحميل النموذج
        tf.keras.models.load_model(file_path)
        return {"status": "valid", "message": "The model file is valid."}
    except (OSError, ValueError) as e:
        # التعامل مع الأخطاء التي تشير إلى ملف تالف أو غير مكتمل
        return {"status": "invalid", "message": f"The model file is corrupted or incomplete: {str(e)}"}

def get_file_info(file_path):
    """
    تطبع اسم الملف، نوعه، وحجمه.
    """
    if os.path.exists(file_path):
        file_name = os.path.basename(file_path)
        file_type = os.path.splitext(file_name)[1]  # امتداد الملف
        file_size = os.path.getsize(file_path)  # الحجم بالبايت
        print(f"اسم الملف: {file_name}")
        print(f"نوع الملف: {file_type or 'غير محدد'}")  # عرض 'غير محدد' إذا لم يكن هناك امتداد
        print(f"حجم الملف: {file_size} بايت")
    else:
        print(f"✘ الملف غير موجود: {file_path}")

def get_file_permissions(file_path):
    """
    تعرض أذونات الملف.
    """
    try:
        file_info = os.stat(file_path)
        # استخدام stat لتحليل الأذونات
        permissions = stat.S_IMODE(file_info.st_mode)
        return permissions
    except Exception as e:
        print(f"Error checking permissions: {e}")
        return None

def check_file_integrity(file_path, expected_size):
    """
    تحقق من تكامل الملف بناءً على الحجم المتوقع.
    """
    try:
        # تحقق من وجود الملف
        if not os.path.exists(file_path):
            print("File does not exist.")
            return False

        # تحقق من حجم الملف
        actual_size = os.path.getsize(file_path)
        if actual_size != expected_size:
            print(f"File size mismatch: expected {expected_size} bytes, but got {actual_size} bytes.")
            return False
        
        # تحقق من صحة النموذج باستخدام TensorFlow
        try:
            model = tf.keras.models.load_model(file_path)
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    except Exception as e:
        print(f"Error in file integrity check: {e}")
        return False

# مسار الملف
file_path = '/app/end3.keras'
expected_size = 369520000  # الحجم المتوقع بالبايت (في حالة الحجم الفعلي من GitHub)

result = check_model_file(model_path)
print("tensorflow say: ",result)
get_file_info(model_path)

# تحقق من صلاحية الأذونات
permissions = get_file_permissions(file_path)
if permissions is None:
    print("Error checking file permissions.")
else:
    print(f"File permissions: {oct(permissions)}")

# التحقق من تكامل الملف
if check_file_integrity(file_path, expected_size):
    print("The file is valid and accessible.")
else:
    print("The file is invalid or inaccessible.")



categories = ['Apple___healthy', 'Apple___Black_rot']

# كود المعالجة السابقه مفروض يتعدل على حسب أخر نموذج (معلوووومه مهههههمه) لازم تتعدل
def preprocess_image(image):
    image = image.resize((256, 256))  # تغيير حجم الصورة
    image_array = np.array(image) / 255.0  # تطبيع القيم
    return image_array.reshape(1, 256, 256, 3)  # إعادة تشكيل البيانات

@app.route('/predict', methods=['POST'])
def predict():
    print("Request method:", request.method)
    print("Request files:", request.files)


    if 'file' not in request.files:
        return jsonify({"error": "لم يتم إرسال ملف"}), 400
    
    file = request.files['file']

    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "نوع الملف غير مدعوم. يرجى تحميل صورة."}), 400

    image = Image.open(file.stream)  # فتح الصورة
    preprocessed_image = preprocess_image(image)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import os

# app = Flask(__name__)
# CORS(app)

# MODEL_PATH = "end3.keras"
# IMG_SIZE = 256

# # تحقق من وجود النموذج أثناء بدء التشغيل
# # if not os.path.exists(MODEL_PATH):
# #     raise FileNotFoundError("النموذج غير موجود! تأكد من رفع الملف قبل تشغيل التطبيق.")

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")


# try:
#     model = tf.keras.models.load_model(MODEL_PATH)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     raise

# # قائمة الفئات
# categories = ['Apple___healthy', 'Apple___Black_rot']
# # class_labels = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"]

# def preprocess_image(image):
#     try:
#         image = image.resize((IMG_SIZE, IMG_SIZE))  # تغيير حجم الصورة
#         image_array = np.array(image) / 255.0  # تطبيع القيم
#         if image_array.shape[-1] != 3:
#             raise ValueError("الصورة ليست بتنسيق RGB")
#         return image_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)  # إعادة تشكيل البيانات
#     except Exception as e:
#         raise ValueError("خطأ في معالجة الصورة: " + str(e))

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "لم يتم إرسال ملف الصورة. تأكد من إرسال صورة بصيغة مدعومة (PNG أو JPG أو JPEG)."}), 400
    
#     file = request.files['file']
#     if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
#         return jsonify({"error": "نوع الملف غير مدعوم. يرجى تحميل صورة."}), 400

#     try:
#         image = Image.open(file.stream)
#         image.verify()  # التحقق من أن الملف هو صورة حقيقية
#         image = Image.open(file.stream)  # إعادة فتح الصورة إذا كانت صالحة
#     except Exception:
#         return jsonify({"error": "الملف ليس صورة صالحة"}), 400

#     try:
#         preprocessed_image = preprocess_image(image)
#     except ValueError as e:
#         return jsonify({"error": str(e)}), 400

#     # إجراء التنبؤ
#     prediction = model.predict(preprocessed_image)
#     predicted_class = int(prediction[0][0] > 0.5)  # تحويل الاحتمالية إلى 0 أو 1
#     confidence = float(prediction[0][0]) if predicted_class == 1 else float(1 - prediction[0][0])

#     # إعداد الاستجابة
#     response = {
#         "predicted_label": categories[predicted_class],
#         "confidence": confidence
#     }

#     return jsonify(response)
#     # predictions = model.predict(preprocessed_image)
#     # class_index = np.argmax(predictions, axis=1)[0]
#     # confidence = np.max(predictions)

#     # response = {
#     #     "class_index": int(class_index),
#     #     "class_label": class_labels[class_index],
#     #     "confidence": float(confidence)
#     # }
#     # return jsonify(response)

# if __name__ == '__main__':
#     app.run()
