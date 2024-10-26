import cv2
from ultralytics import YOLO
import csv
from datetime import datetime

# Load the YOLOv8 model
model = YOLO("yolo11n.pt")

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Open a CSV file to log detections
with open('detections.csv', mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header if the file is empty
    csv_writer.writerow(['Detection'])

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Run the YOLO model on the frame
            results = model(frame)

            # Check for detected objects
            if results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:  # Check if boxes are detected
                detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time
                # Write to the CSV file
                csv_writer.writerow([f"Object detected at time {detection_time}"])
                print(f"Object detected at time {detection_time}")  # Print detection message

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
