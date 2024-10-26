import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform inference
        results = model(frame)

        # Extract the detected objects
        detected_objects = []
        for result in results:
            if result.boxes:  # Check if boxes are detected
                for box in result.boxes:
                    cls = int(box.cls)  # Get class ID
                    conf = box.conf  # Get confidence score
                    if conf > 0.5:  # Only consider detections with confidence > 0.5
                        detected_objects.append(model.names[cls])  # Append class name

        # Print detected objects to the terminal
        if detected_objects:
            print("Detected objects:", set(detected_objects))  # Use set to avoid duplicates

        # Display the resulting frame (optional)
        cv2.imshow('YOLOv8 Object Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    # Handle Ctrl+C gracefully
    print("Exiting...")

finally:
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
