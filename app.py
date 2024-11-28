import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load model (adjust the path to your model file)
model = load_model('plant_disease_model.keras')

# Define the allowed extensions for the uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Set up upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if file is a valid image
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing function
def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))  # Assuming model input size is 224x224
    img_normalized = img_resized / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    return img_normalized

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the image for prediction
        processed_image = process_image(file_path)

        # Get model prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions)

        # Here, you need to define class labels based on your dataset
        class_labels = ['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy']  # Example labels
        prediction_result = class_labels[predicted_class]

        # Send the result to result.html
        return render_template('result.html', prediction=prediction_result, img_path='uploads/' + filename)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
