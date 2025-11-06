import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from datetime import datetime
from facedb import FaceDB

# Ініціалізація Flask додатку
app = Flask(__name__)
app.secret_key = "your_secret_key_here_12345"

# Ініціалізація FaceDB
db = FaceDB(
    path="facedata",
    metric="euclidean",
    embedding_dim=128,
    module="face_recognition"
)

print("="*50)
print("🚀 Сервер розпізнавання облич запущено!")
print("📂 База даних: facedata/")
print("="*50)


@app.route("/", methods=["GET", "POST"])
def add_face_data():
    """
    Маршрут для головної сторінки та додавання облич до бази даних
    """
    if request.method == "POST":
        try:
            # Отримання імені з форми
            name = request.form.get("name", "").strip()
            
            if not name:
                return jsonify({
                    "message": "Помилка: Ім'я не може бути порожнім",
                    "category": "error"
                }), 400
            
            # Отримання файлу зображення
            img_file = request.files.get("image")
            
            if not img_file or img_file.filename == "":
                return jsonify({
                    "message": "Помилка: Файл зображення не обрано",
                    "category": "error"
                }), 400
            
            # Читання вмісту файлу
            img_bytes = img_file.read()
            
            if len(img_bytes) == 0:
                return jsonify({
                    "message": "Помилка: Файл зображення порожній",
                    "category": "error"
                }), 400
            
            # Додавання обличчя до бази даних
            print(f"➕ Додавання нового обличчя: {name}")
            face_id = db.add(name, img=img_bytes)
            
            print(f"✅ Успішно додано: {name} (ID: {face_id})")
            
            return jsonify({
                "message": f"Успішно додано: {name}",
                "category": "success",
                "face_id": face_id
            }), 200
            
        except ValueError as e:
            error_msg = str(e)
            if "No face detected" in error_msg:
                print(f"❌ Помилка: Обличчя не знайдено на фото")
                return jsonify({
                    "message": "Помилка: Обличчя не знайдено на фотографії. Переконайтеся, що обличчя добре видно.",
                    "category": "error"
                }), 400
            else:
                print(f"❌ ValueError: {error_msg}")
                return jsonify({
                    "message": f"Помилка: {error_msg}",
                    "category": "error"
                }), 400
                
        except Exception as e:
            print(f"❌ Несподівана помилка: {str(e)}")
            return jsonify({
                "message": f"Помилка сервера: {str(e)}",
                "category": "error"
            }), 500
    
    # GET запит - показати головну сторінку
    return render_template("index.html")


@app.route("/recognize", methods=["GET", "POST"])
def recognize_face():
    """
    Маршрут для розпізнавання облич
    """
    if request.method == "POST":
        try:
            # Отримання файлу зображення
            img_file = request.files.get("image")
            
            if not img_file or img_file.filename == "":
                return jsonify({
                    "message": "Помилка: Файл зображення не обрано",
                    "category": "error"
                }), 400
            
            # Читання вмісту файлу
            img_bytes = img_file.read()
            
            if len(img_bytes) == 0:
                return jsonify({
                    "message": "Помилка: Файл зображення порожній",
                    "category": "error"
                }), 400
            
            # Конвертація байтів у формат OpenCV
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({
                    "message": "Помилка: Не вдалося прочитати зображення. Файл може бути пошкоджений.",
                    "category": "error"
                }), 400
            
            # Підготовка зображення для FaceDB (конвертація BGR -> RGB)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Розпізнавання обличчя
            print("🔍 Розпізнавання обличчя...")
            result = db.recognize(img=rgb_img, include=["name", "confidence"])
            
            # Аналіз результату
            if result and result.id:
                confidence = result.confidence
                name = result.name
                
                print(f"✅ Розпізнано: {name} (впевненість: {confidence:.2%})")
                
                return jsonify({
                    "name": name,
                    "confidence": confidence,
                    "category": "success",
                    "message": f"Розпізнано: {name}"
                }), 200
            else:
                print("⚠️ Обличчя не розпізнано (немає в базі)")
                return jsonify({
                    "name": None,
                    "message": "Невідома особа. Ця людина відсутня в базі даних.",
                    "category": "warning"
                }), 200
                
        except ValueError as e:
            error_msg = str(e)
            if "No face detected" in error_msg:
                print(f"❌ Помилка: Обличчя не знайдено на фото")
                return jsonify({
                    "message": "Помилка: Обличчя не знайдено на фотографії. Переконайтеся, що обличчя добре видно.",
                    "category": "error"
                }), 400
            else:
                print(f"❌ ValueError: {error_msg}")
                return jsonify({
                    "message": f"Помилка: {error_msg}",
                    "category": "error"
                }), 400
                
        except Exception as e:
            print(f"❌ Несподівана помилка: {str(e)}")
            return jsonify({
                "message": f"Помилка сервера: {str(e)}",
                "category": "error"
            }), 500
    
    # GET запит - показати головну сторінку
    return render_template("index.html")


if __name__ == "__main__":
    # Створення папки facedata, якщо не існує
    if not os.path.exists("facedata"):
        os.makedirs("facedata")
        print("📁 Створено папку facedata/")
    
    # Запуск сервера
    print("\n🌐 Відкрийте браузер та перейдіть за адресою:")
    print("   http://127.0.0.1:5000\n")
    
    app.run(debug=True, host="127.0.0.1", port=5000)