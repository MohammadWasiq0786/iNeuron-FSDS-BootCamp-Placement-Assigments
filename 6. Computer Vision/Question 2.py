"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Computer Vision Assignment 2
"""

"""
Q2. - From Question 1, you would get a trained model which would classify the vegetables based on the classes. You need to convert the trained model to ONNX format and achieve faster inference
Note -
1. There is no set inference time, but try to achieve as low an inference time as
possible
2. Create a web app to interact with the model, where the user can upload the
image and get predictions
3. Try to reduce the model size considerably so that inference time can be faster
4. Use modular Python scripts to train and infer the model
5. Only Jupyter notebooks will not be allowed
6. Write code comments whenever needed for understanding
"""

# Ans:

import torch
import torchvision.models as models
import torch.onnx as onnx

# Step 1: Train the Model 
model = models.resnet18(pretrained=True)


# Step 2: Convert the Model to ONNX Format
# Load the trained model
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Example input tensor
example_input = torch.randn(1, 3, 224, 224)

# Convert the model to ONNX format
onnx_path = 'model.onnx'
torch.onnx.export(model, example_input, onnx_path, export_params=True)

print(f"Model converted and saved as '{onnx_path}'")

import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load the image and apply necessary transformations
image_path = 'test_image.jpg'
image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_data = preprocess(image)
input_data = input_data.unsqueeze(0)  # Add batch dimension

# Load the ONNX model
onnx_model = onnxruntime.InferenceSession('model.onnx')

# Run the inference
input_name = onnx_model.get_inputs()[0].name
output_name = onnx_model.get_outputs()[0].name
output = onnx_model.run([output_name], {input_name: input_data.numpy()})

# Get the predicted class
predicted_class = np.argmax(output[0])

print(f"Predicted class: {predicted_class}")

from flask import Flask, render_template, request
import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image_path = 'uploaded_image.jpg'
    image.save(image_path)

    # Load and preprocess the image
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_data = preprocess(image)
    input_data = input_data.unsqueeze(0)  # Add batch dimension

    # Load the ONNX model
    onnx_model = onnxruntime.InferenceSession('model.onnx')

    # Run the inference
    input_name = onnx_model.get_inputs()[0].name
    output_name = onnx_model.get_outputs()[0].name
    output = onnx_model.run([output_name], {input_name: input_data.numpy()})

    # Get the predicted class
    predicted_class = np.argmax(output[0])

    return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run()