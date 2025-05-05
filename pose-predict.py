import cv2
from ultralytics import YOLO
import numpy as np
import os
import json

model_a = YOLO('runs/train-pose/train55/weights/best.pt')
model_b = YOLO('runs/train-pose/train52/weights/best.pt')

def detect_objects(image_path):
    img = cv2.imread(image_path)
    results = model_a(img, iou=0.4)
    cropped_images = []
    expanded_coords = []
    i=0
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for i,box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            x1_expanded = max(x1 - 10, 0)
            y1_expanded = max(y1 - 10, 0)
            x2_expanded = min(x2 + 10, img.shape[1])
            y2_expanded = min(y2 + 10, img.shape[0])
            cropped_img = img[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
            cropped_images.append(cropped_img)
            cv2.imwrite(f'tmpsave/{i}.jpg', cropped_img)
            expanded_coords.append((x1_expanded, y1_expanded, x2_expanded, y2_expanded))

    return cropped_images, expanded_coords

def detect_keypoints(cropped_images, expanded_coords, original_img):
    all_keypoints = []
    model_b.predict(cropped_images,save=False,conf=0.6,show_labels=False)
    for cropped_img, (x1, y1, x2, y2) in zip(cropped_images, expanded_coords):
        results = model_b(cropped_img,conf=0.6)
        boxes = results[0].boxes.xyxy.cpu().numpy()[0] 
        print(boxes)
        print(results[0].keypoints.xy.cpu().numpy())
        keypoints = results[0].keypoints.xy.cpu().numpy()[0]
        confidences = results[0].boxes.conf.cpu().numpy() 
        box_mapped = boxes[:4] + np.array([x1, y1, x1, y1])
        keypoints_mapped = []
        for i,keypoint in enumerate(keypoints):
            keypoint_map = keypoint + np.array([x1, y1])
            if i!=0:
                cv2.circle(original_img, (int(keypoint_map[0]), int(keypoint_map[1])),3, (0, 255, 0), -1)
            else:
                cv2.circle(original_img, (int(keypoint_map[0]), int(keypoint_map[1])),3, (0, 0, 255), -1)
            keypoints_mapped.append(keypoint_map)
        all_keypoints.append((box_mapped, keypoints_mapped))
        cv2.rectangle(original_img, (int(box_mapped[0]), int(box_mapped[1])), 
                          (int(box_mapped[2]), int(box_mapped[3])), (255, 0, 0), 2) 
        cv2.putText(original_img, f'{confidences[0]:.2f}', (int(box_mapped[0]), int(box_mapped[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return original_img, all_keypoints

def convert_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data

def process_image(image_path):
    file_name = os.path.basename(image_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    output_file_name = file_name_without_extension + "_pose.png"
    output_path = os.path.join("final", output_file_name)
    original_img = cv2.imread(image_path)
    cropped_images, expanded_coords = detect_objects(image_path)
    processed_img, keypoints = detect_keypoints(cropped_images, expanded_coords, original_img)

    os.makedirs("final", exist_ok=True)

    cv2.imwrite(output_path, processed_img)


for file in os.listdir("autodl-tmp/test"):
    if file.endswith(".png"):
        process_image(os.path.join("autodl-tmp/test", file))