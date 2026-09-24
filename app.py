from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# ----------------------------
# Flask Configuration
# ----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ----------------------------
# Model and Parameters
# ----------------------------
MODEL_PATH = 'effnetb3_cancer_final.keras'  # model file in same directory
IMG_SIZE = (300, 300)
CLASS_NAMES = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']

# ----------------------------
# Load Model Once
# ----------------------------
print("🔍 Loading model from:", MODEL_PATH)
if os.path.isdir(MODEL_PATH):  # if it's a SavedModel directory
    model = load_model(MODEL_PATH, compile=False)
else:
    model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# ----------------------------
# Prediction Function
# ----------------------------
def predict_image(img_path):
    # Preprocess image same as Colab
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0])) * 100
    pred_label = CLASS_NAMES[pred_idx]
    return pred_label, confidence

# ----------------------------
# Routes
# ----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_filename = None
    error = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            error = "Please select an image file."
        else:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            # Predict
            pred_label, conf = predict_image(img_path)
            prediction = pred_label
            confidence = f"{conf:.2f}%"
            img_filename = file.filename

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        image_file=img_filename,
        error=error
    )

# ----------------------------
# Run Flask App
# ----------------------------
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
