# GreenClassify: Flask Web Application for Vegetable Classification

from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
try:
    model = load_model('vegetable_classification.h5')
except FileNotFoundError:
    print("Model file 'vegetable_classification.h5' not found. Please train the model first using train.py.")
    model = None

# Class names (must match the training class_indices order)
class_names = [
    'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
    'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
    'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
]  # Adjust based on actual dataset classes

# Ensure uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess the image
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize

            # Predict
            if model is None:
                prediction = "Model not loaded. Please train the model first."
            else:
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])
                prediction = class_names[predicted_class_index]

            # Render prediction page with result
            return render_template('prediction.html', prediction=prediction)

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
