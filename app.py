from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Load your pre-trained model
model_path = 'new/model_name.h5'
model = load_model(model_path)

# Define class labels
class_labels = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

# Ensure the upload directory exists
UPLOAD_FOLDER = 'new/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(file_path)

        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict the class probabilities
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]

        return render_template("Acknowledgement.html", name=f.filename, label=predicted_class_label, img_path=file_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
