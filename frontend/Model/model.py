import torch
import coremltools as ct
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

print("Create input")
# Create a dummy input for conversion (1 batch size, 3 channels, 640x640 size)
dummy_input = torch.rand(1, 3, 640, 640)

print("Convert to ML format")
# Convert to Core ML format
ml_model = ct.convert(
    model,
    inputs=[ct.ImageType(shape=(1, 3, 640, 640))],  # Adjust the shape if necessary
    convert_to="mlprogram",  # You can also use "neuralnetwork" depending on your use case
    source='pytorch'  # Specify that the model is from PyTorch
)

# Save the Core ML model
ml_model.save('yolov8n.mlmodel')

print("Model converted and saved as yolov8n.mlmodel")
