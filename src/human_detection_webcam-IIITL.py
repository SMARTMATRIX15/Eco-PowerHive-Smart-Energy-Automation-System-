import cv2
import serial
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from torchvision import models
import clip
import mediapipe as mp
import numpy as np
import json

with open("config.json", "r") as f:
    config = json.load(f)

print("Energy Saving Threshold:", config["thresholds"]["energy_saving_mode"])


# Initialize YOLOv8 model
model_yolo = YOLO("yolov8m.pt")

# Initialize Vision Transformer (ViT) model
model_vit = models.vit_b_16(pretrained=True)
model_vit.eval()

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Transformation for ViT input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize serial communication with Arduino
arduino = serial.Serial('/dev/tty.usbserial-130', 9600)
time.sleep(2)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

# Store previous pose landmarks
prev_landmarks = None

# Temporal filter variables
motion_buffer = []  # Stores motion detection status across multiple frames
motion_buffer_size = 5  # Number of frames to check consistency for motion detection
person_buffer = []  # Stores person detection status across multiple frames
person_buffer_size = 5  # Number of frames to check consistency for person detection

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolo(frame)
        detections = results[0].boxes.data
        person_detected = False
        motion_detected = False

        for det in detections:
            class_id = int(det[5])
            if class_id == 0:  # Person class ID in YOLO
                person_detected = True
                x1, y1, x2, y2 = map(int, det[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                # Process the person image for ViT
                person_crop_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(person_crop_pil).unsqueeze(0)

                with torch.no_grad():
                    vit_output = model_vit(input_tensor)
                    _, predicted_class = torch.max(vit_output, 1)

                # Use CLIP model to determine if it's a person or not
                image_clip = preprocess_clip(person_crop_pil).unsqueeze(0).to(device)
                text_inputs = clip.tokenize(["a person", "not a person"]).to(device)
                with torch.no_grad():
                    image_features = model_clip.encode_image(image_clip)
                    text_features = model_clip.encode_text(text_inputs)
                    similarity = (image_features @ text_features.T).softmax(dim=-1)
                    clip_prediction = "Person" if similarity[0, 0] > similarity[0, 1] else "Not a Person"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{clip_prediction} - {predicted_class.item()}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Pose detection for motion detection
                person_crop_resized = cv2.resize(person_crop, (256, 256))  
                person_crop_rgb = cv2.cvtColor(person_crop_resized, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(person_crop_rgb)

                if results_pose.pose_landmarks:
                    height, width, _ = person_crop.shape  
                    current_landmarks = np.array([[lm.x * width, lm.y * height] for lm in results_pose.pose_landmarks.landmark])
                    
                    if prev_landmarks is not None:
                        motion = np.linalg.norm(current_landmarks - prev_landmarks, axis=1).mean()
                        if motion > 5:  # Threshold for movement
                            motion_detected = True
                    
                    prev_landmarks = current_landmarks  # Store landmarks for next frame comparison
                    
                    for landmark in results_pose.pose_landmarks.landmark:
                        x = int(x1 + landmark.x * width)
                        y = int(y1 + landmark.y * height)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                    mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
        
        # Temporal filtering for motion detection
        motion_buffer.append(motion_detected)
        if len(motion_buffer) > motion_buffer_size:
            motion_buffer.pop(0)
        consistent_motion = sum(motion_buffer) >= motion_buffer_size * 0.8  # Require 80% consistent detection

        # Temporal filtering for person detection
        person_buffer.append(person_detected)
        if len(person_buffer) > person_buffer_size:
            person_buffer.pop(0)
        consistent_person = sum(person_buffer) >= person_buffer_size * 0.8  # Require 80% consistent detection

        # Send signal to Arduino based on consistent motion and person detection
        if consistent_person and consistent_motion:
            arduino.write(b'1')
        else:
            arduino.write(b'0')

        cv2.imshow("Enhanced YOLOv8 + ViT + CLIP + Pose + Activity Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()