from flask import Flask, request, render_template, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import csv

app = Flask(__name__)

# Load the class names and descriptions from the CSV file
class_info = {}
with open('food-des.csv', 'r', encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    for i, row in enumerate(csvreader):
        class_info[i] = (row[0], row[1])

# Load your trained state_dict (adjust this as needed for your model)
state_dict = torch.load('model_state_dict.pth', map_location=torch.device('cpu'))
num_classes = 30
# Check the keys in the state_dict to determine the architecture
architecture = 'unknown'
if 'classifier.1.weight' in state_dict:
    architecture = 'mobilenetv2'
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
elif 'fc.weight' in state_dict:
    architecture = 'resnet18'
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
else:
    raise ValueError('Unknown architecture in the state_dict.')

# Load the state_dict
model.load_state_dict(state_dict)
model.eval()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('static/uploads', filename)
        file.save(filepath)

        prediction, description, confidence = make_prediction(filepath)

        prediction_image_url = url_for('static', filename='uploads/' + filename)
        return render_template('results.html', prediction=prediction, description=description, confidence=confidence, prediction_image_url=prediction_image_url)
    else:
        return render_template('index.html', error='Invalid file type')

@app.route('/delete-file', methods=['GET'])
def delete_file():
    filename = request.args.get('filename')
    if filename:
        filepath = os.path.join('static/uploads', filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"success": True}), 200
        else:
            return jsonify({"error": "File not found"}), 404
    else:
        return jsonify({"error": "No filename provided"}), 400

def make_prediction(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    
    # Resize image, center crop, and normalize
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),  # Adjust the size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        max_prob, predicted = torch.max(probabilities, 0)

    # Retrieve the class name and description using the predicted index
    class_idx = predicted.item()
    class_name, description = class_info[class_idx]
    confidence = round(max_prob.item(), 2)

    return class_name, description, confidence

if __name__ == '__main__':
    app.run(debug=True)
