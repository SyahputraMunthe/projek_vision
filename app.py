import os
import joblib
import gdown

MODEL_PATH = "model_svm_tbc.pkl"

FILE_ID = "1mclt8gX9GA2DUbtaing_J0AUwRXRGVMk"

URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(URL, MODEL_PATH, quiet=False)

print("Loading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded.")
