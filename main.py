import cv2
import numpy as np


try:
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
except:
    print("Error: YOLO model files missing! Ensure 'yolov3.weights' & 'yolov3.cfg' exist.")
    exit()


try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except:
    print("Error: Could not load 'coco.names'. Ensure the file exists.")
    exit()

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, _ = frame.shape


    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5: 
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype("int")

                x = max(0, int(center_x - w / 2))
                y = max(0, int(center_y - h / 2))

                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"

               
                cv2.putText(frame, label, (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
              

    cv2.imshow('Object Detection', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
