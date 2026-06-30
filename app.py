from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
import pickle
import gdown

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
MODEL_PATH = "model_svm_tbc.pkl"

FILE_ID = "1mclt8gX9GA2DUbtaing_J0AUwRXRGVMk"

URL = f"https://drive.google.com/uc?id=1mclt8gX9GA2DUbtaing_J0AUwRXRGVMk"

if not os.path.exists(MODEL_PATH):
    print("Mengunduh model...")
    gdown.download(URL, MODEL_PATH, quiet=False)

print("Model ada?", os.path.exists(MODEL_PATH))
print("Ukuran model:", os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model berhasil dimuat.")


def preprocess(image_path):
    img = cv2.imread(image_path)

    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0

    feature = gray.flatten().reshape(1, -1)

    return feature


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    if file.filename == "":
        return "Tidak ada file"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    feature = preprocess(filepath)

    prediction = model.predict(feature)[0]

    if prediction == 0:
        result = "Normal"
    else:
        result = "Tuberkulosis (TBC)"

    return render_template(
        "result.html",
        result=result,
        image=file.filename
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
