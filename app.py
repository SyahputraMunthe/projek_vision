from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==========================
# Download dan Load Model
# ==========================
MODEL_PATH = "model_svm_tbc.pkl"

print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded.")



# ==========================
# Preprocessing
# ==========================

def preprocess(image_path):
    img = cv2.imread(image_path)

    img = cv2.resize(img, (224, 224))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = gray.astype("float32") / 255.0

    feature = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    feature = feature.reshape(1, -1)

    return feature


# ==========================
# Routing
# ==========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    if file.filename == "":
        return "Tidak ada file dipilih"

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


# ==========================
# Run Local
# ==========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
