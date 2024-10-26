import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Set the model to evaluation mode
model.eval()

# Create a dummy input for TorchScript (1 batch size, 3 channels, 640x640 size)
dummy_input = torch.rand(1, 3, 640, 640)

# Export the model to TorchScript
traced_model = torch.jit.trace(model.model, dummy_input)  # Accessing the internal model

# Save the TorchScript model
traced_model.save('yolov8_traced.pt')

print("TorchScript model saved as yolov8_traced.pt")
