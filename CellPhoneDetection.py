import cv2
import numpy as np
from ultralytics import YOLO
import math  # Import the math module

# Define a threshold for the proximity to determine if a person is holding a phone
PHONE_PROXIMITY_THRESHOLD = 100  # You can adjust this value based on your requirements

model = YOLO("yolov8m.pt")  # Or a smaller model like yolov8n.pt for faster processing

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

cv2.namedWindow("Video - Detected Persons and Phones", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video - Detected Persons and Phones", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_resized = cv2.resize(frame, (640, 480))

    results = model(frame_resized)

    person_boxes = []
    phone_boxes = []

    # Iterate over the results. The results can be a list of detections
    for result in results:
        # Extract the boxes and classes for each detection
        boxes = result.boxes.xyxy  # Bounding box coordinates (xyxy)
        classes = result.boxes.cls  # Class IDs

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())  # Convert box coordinates to integers

            if classes[i] == 0:  # Person class ID
                person_boxes.append((box, i))
            elif classes[i] == 67:  # Cell Phone class ID
                phone_boxes.append((box, i))

    # Phone Detection Logic
    for person_box, person_idx in person_boxes:
        x1, y1, x2, y2 = map(int, person_box.cpu().numpy())  # Convert person box to integer coordinates

        # Initially, draw a green bounding box around the person
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for person

        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2

        for phone_box, phone_idx in phone_boxes:
            px1, py1, px2, py2 = map(int, phone_box.cpu().numpy())  # Convert phone box to integer coordinates

            phone_center_x = (px1 + px2) // 2
            phone_center_y = (py1 + py2) // 2

            # Calculate the distance between person and phone
            distance = math.dist((person_center_x, person_center_y), (phone_center_x, phone_center_y))

            if distance < PHONE_PROXIMITY_THRESHOLD:
                # If person is holding the phone, draw a red bounding box around both
                cv2.rectangle(frame_resized, (px1, py1), (px2, py2), (0, 0, 255), 2)  # Red for phone
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for person as well
                cv2.putText(frame_resized, "Holding Phone", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 2)  # Text indicating holding a phone

    cv2.imshow("Video - Detected Persons and Phones", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Video - Detected Persons and Phones", cv2.WND_PROP_VISIBLE) < 1:
        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
