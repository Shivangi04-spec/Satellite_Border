import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
MODEL_PATH = "data/satellite_cnn_model.h5"
model = load_model(MODEL_PATH)
class_names = ['buildings', 'land', 'roads', 'vegetation', 'water']
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image. Check image format.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image
def predict_image(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    return predicted_class, confidence
TEST_IMAGE = "data/test/roads/Highway_6.jpg"
predicted_class, confidence = predict_image(TEST_IMAGE)
print("Predicted Terrain:", predicted_class)
print(f"Confidence: {confidence:.2f}%")
