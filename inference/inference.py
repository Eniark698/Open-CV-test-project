import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load environment variables
INPUT_FOLDER = os.getenv('INPUT_FOLDER', './inference/input/')
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', './inference/output/')
MODEL_PATH = os.getenv('MODEL_PATH', './model/unet_model.keras')

# Define global variables
IMG_SIZE = (256, 256)


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


# Function to preprocess the image
def preprocess_image(image_path, img_size=IMG_SIZE):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to postprocess the prediction mask
def postprocess_mask(mask, original_size):
    mask = np.squeeze(mask)  # Remove batch and channel dimensions
    mask = (mask > 0.5).astype(np.uint8)  # Threshold to binary
    mask = cv2.resize(mask, original_size)
    return mask

# Load the trained model
model = load_model(MODEL_PATH, custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient})

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Process each image in the input folder
for image_name in os.listdir(INPUT_FOLDER):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(INPUT_FOLDER, image_name)

        # Preprocess the image
        image = preprocess_image(image_path)

        # Perform inference
        prediction = model.predict(image)

        # Postprocess the mask
        original_size = cv2.imread(image_path).shape[:2]
        mask = postprocess_mask(prediction, original_size)

        # Save the mask
        output_path = os.path.join(OUTPUT_FOLDER, f'{os.path.splitext(image_name)[0]}_mask.png')
        cv2.imwrite(output_path, mask * 255)  # Multiply by 255 to save as an image

print("Inference completed. Masks saved to:", OUTPUT_FOLDER)
