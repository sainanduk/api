from flask import Flask, request, jsonify
from tensorflow.keras.models import model_from_json
import numpy as np
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load your model architecture and weights
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model_weights.h5')

# Load class labels
class_labels = ["NO_TUMOR", "TUMOR"]

# Function to preprocess the image
def preprocess_image(image):
    # Convert image to grayscale
    gray_img = image.convert('L')
    
    # Resize image to match model input shape (assuming it's 200x200)
    resized_img = gray_img.resize((200, 200))
    
    # Convert image to numpy array
    img_array = np.array(resized_img)
    
    # Expand dimensions to add batch size dimension
    input_image = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    
    # Normalize pixel values (optional, if not done during training)
    input_image = input_image / 255.0  # Assuming pixel values are in the range [0, 255]
    
    # Reshape image to match model input shape
    input_image = np.expand_dims(input_image, axis=0)
    
    return input_image

# Function to load image from URL
def load_image_from_url(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image

# Function to predict the class of the image
def predict_image(image_url):
    # Load image from URL
    image = load_image_from_url(image_url)
    
    # Preprocess the image
    input_image = preprocess_image(image)
    
    # Make predictions on the preprocessed image
    predictions = loaded_model.predict(input_image)
    
    # Convert the predictions to a more interpretable format
    rounded_predictions = np.round(predictions, decimals=2)
    
    # Assuming binary classification
    pred_class_index = np.argmax(rounded_predictions)
    pred_class_label = class_labels[pred_class_index]
    
    return pred_class_label, rounded_predictions[0][pred_class_index]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image URL from the request
        image_url = request.json.get('image_url')

        # Check if the URL is provided
        if not image_url:
            return jsonify({'error': 'No image URL provided'})

        # Call the predict_image function with the image URL
        pred_class_label, probability = predict_image(image_url)

        # Return the prediction as JSON
        return jsonify({'predicted_class': pred_class_label, 'probability': float(probability)})

if __name__ == '__main__':
    app.run(debug=True)
