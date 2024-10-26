import coremltools as ct
import requests
from PIL import Image
from io import BytesIO
import numpy as np

# Load the CoreML model
model = ct.models.MLModel("yolo11n.mlpackage")

# Download the image from the URL
image_url = "https://ultralytics.com/images/bus.jpg"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content)).convert("RGB")

# Resize the image to the model input size (if needed)
img_resized = img.resize((640, 640))  # Adjust size based on model input requirements

# Prepare input dictionary
input_data = {"image": img_resized}  # Use the resized PIL image

# Run inference
predictions = model.predict(input_data)

# Extract and interpret predictions
output = predictions['var_1227']  # Adjust the key based on your model's output
print("Output shape:", output.shape)  # Check the shape of the output

# Adjust interpretation based on the output shape
if output.ndim == 3:  # If output is 3D
    batch_size = output.shape[0]
    num_predictions = output.shape[1]  # Assuming predictions are in the second dimension
    # Modify this part based on what the last dimension represents
    boxes = output[..., :4]  # Assuming first four elements are bounding boxes
    confidences = output[..., 4]  # Assuming fifth element is confidence score
    class_scores = output[..., 5:]  # Remaining are class probabilities

    # Calculate final class scores
    final_scores = confidences[..., np.newaxis] * class_scores

    # Get predicted class and confidence score
    predicted_classes = np.argmax(final_scores, axis=-1)
    predicted_confidences = np.max(final_scores, axis=-1)

    # Example: Print out predictions
    for i in range(batch_size):
        for j in range(num_predictions):
            if predicted_confidences[i, j] > 0.5:  # Confidence threshold
                x_center, y_center, width, height = boxes[i, j]
                print(f"Detection {j}: Class {predicted_classes[i, j]} with confidence {predicted_confidences[i, j]:.2f} at ({x_center}, {y_center}) with width {width} and height {height}")

else:
    print("Unexpected output shape. Please check the model output.")
