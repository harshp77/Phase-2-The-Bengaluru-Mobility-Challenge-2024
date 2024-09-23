import cv2
import pandas as pd
from tqdm import tqdm
from extract_feat import extract_features
import numpy as np
from collections import Counter

def trak_and_extract_features(model, video_path):
    class_labels = ["Bicycle", "Bus", "Car", "LCV", "Three-Wheeler", "Truck", "Two-Wheeler"]

    cap = cv2.VideoCapture(video_path)
    features_dict = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    
    # Dictionary to store the features, class labels, bounding box areas, and bbox for each track_id
    running_features = {}

    with tqdm(total=total_frames, desc="Processing Video") as pbar:
        while cap.isOpened():
            # Read the next frame from the video
            success, frame = cap.read()
            if success:
                frame_id += 1
                # Perform tracking on the current frame
                results = model.track(frame, persist=True, verbose=False, iou=0.10)
                pbar.update(1)

                if results[0].boxes:
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    track_ids = results[0].boxes.id
                    classes = results[0].boxes.cls

                    if track_ids is not None and classes is not None:
                        track_ids = track_ids.int().cpu().tolist()
                        classes = classes.int().cpu().tolist()

                        # Process each detected object in the frame
                        for box, track_id, cls in zip(boxes, track_ids, classes):
                            class_label = class_labels[cls]

                            # Convert bounding box from center x, y, width, height to top-left x, y, width, height
                            x, y, w, h = [int(coord) for coord in box]
                            area = w * h
                            cropped_image = frame[y-int(h/2):y+int(h/2), x-int(w/2):x+int(w/2)]
                            
                            x_center, y_center, w, h = [int(coord) for coord in box]
                            x = int(x_center - w / 2)  # Calculate top-left x from center x
                            y = int(y_center - h / 2)  # Calculate top-left y from center y

                            # Extract features from the cropped image
                            features = extract_features(cropped_image)

                            # If track_id is new, initialize its entry
                            if track_id not in running_features:
                                running_features[track_id] = {
                                    'first_frame': frame_id,
                                    'class_labels': [],  # Store all class labels for this track_id
                                    'features_list': [],
                                    'area_list': [],
                                    'bbox_list': [],  # Store bounding boxes for each detection
                                    'frame_list': []  # Store frame numbers for each detection
                                }

                            # Append the extracted features, bounding box area, bbox, and class label to the list for this track_id
                            running_features[track_id]['features_list'].append(features)
                            running_features[track_id]['area_list'].append(area)
                            running_features[track_id]['bbox_list'].append((x, y, w, h))
                            running_features[track_id]['frame_list'].append(frame_id)
                            running_features[track_id]['class_labels'].append(class_label)
                            running_features[track_id]['last_frame'] = frame_id
            else:
                print("Completed")
                break

        cap.release()

    # Compute the mean feature vector for each track_id, considering bounding box areas and most frequent class label
    for track_id, track_info in running_features.items():
        if len(track_info['features_list']) > 0:
            # Check if the object has been in the frame for more than 40 frames
            if track_info['last_frame'] - track_info['first_frame'] > 40:
                # Calculate the median area
                sorted_indices = np.argsort(track_info['area_list'])
                median_idx = sorted_indices[len(sorted_indices) // 2]
                median_area = track_info['area_list'][median_idx]

                # Filter features based on bounding box area greater than or equal to the median
                valid_features = [feat for feat, area in zip(track_info['features_list'], track_info['area_list']) if area >= median_area]

                if valid_features:
                    # Calculate the mean of the valid feature vectors
                    mean_features = np.mean(valid_features, axis=0)
                    
                    # Get the most frequent class label
                    most_common_class_label = Counter(track_info['class_labels']).most_common(1)[0][0]

                    # Get the bounding box and frame number corresponding to the median area
                    median_bbox = track_info['bbox_list'][median_idx]
                    median_frame = track_info['frame_list'][median_idx]

                    features_dict[track_id] = {
                        'track_id': track_id,
                        'frame_id': track_info['first_frame'],  # Use the first appearance frame
                        'class_label': most_common_class_label,  # Use the most frequent class label
                        'features': mean_features.tolist(),
                        'median_bbox': median_bbox,  # Bounding box of the median area
                        'median_frame': median_frame  # Frame number of the median area
                    }

    return features_dict


###################################### DEAD CODE #####################################

# def trak_and_extract_features_filter(model, video_path):
#     class_labels = ["Bicycle", "Bus", "Car", "LCV", "Three-Wheeler", "Truck", "Two-Wheeler"]

#     cap = cv2.VideoCapture(video_path)
#     features_dict = {}
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_id = 0
    
#     # Dictionary to store the features and bounding box areas for each track_id
#     running_features = {}

#     with tqdm(total=total_frames, desc="Processing Video") as pbar:
#         while cap.isOpened():
#             # Read the next frame from the video
#             success, frame = cap.read()
#             if success:
#                 frame_id += 1
#                 # Perform tracking on the current frame
#                 results = model.track(frame, persist=True, verbose=False, iou=0.10)
#                 pbar.update(1)

#                 if results[0].boxes:
#                     boxes = results[0].boxes.xywh.cpu().numpy()
#                     track_ids = results[0].boxes.id
#                     classes = results[0].boxes.cls

#                     if track_ids is not None and classes is not None:
#                         track_ids = track_ids.int().cpu().tolist()
#                         classes = classes.int().cpu().tolist()

#                         # Process each detected object in the frame
#                         for box, track_id, cls in zip(boxes, track_ids, classes):
#                             class_label = class_labels[cls]

#                             # Calculate the bounding box dimensions and area
#                             x, y, w, h = [int(coord) for coord in box]
#                             area = w * h
#                             cropped_image = frame[y:y+h, x:x+w]

#                             # Extract features from the cropped image
#                             features = extract_features(cropped_image)

#                             # If track_id is new, initialize its entry
#                             if track_id not in running_features:
#                                 running_features[track_id] = {
#                                     'first_frame': frame_id,
#                                     'class_label': class_label,
#                                     'features_list': [],
#                                     'area_list': []
#                                 }

#                             # Append the extracted features and bounding box area to the list for this track_id
#                             running_features[track_id]['features_list'].append(features)
#                             running_features[track_id]['area_list'].append(area)
#                             running_features[track_id]['last_frame'] = frame_id
#             else:
#                 print("Completed")
#                 break

#         cap.release()

#     # Compute the mean feature vector for each track_id, considering bounding box areas
#     for track_id, track_info in running_features.items():
#         if len(track_info['features_list']) > 0:
#             # Check if the object has been in the frame for more than 40 frames
#             if track_info['last_frame'] - track_info['first_frame'] > 40:
#                 # Calculate the median area
#                 sorted_areas = sorted(track_info['area_list'])
#                 median_area = sorted_areas[len(sorted_areas) // 2]

#                 # Filter features based on bounding box area greater than or equal to the median
#                 valid_features = [feat for feat, area in zip(track_info['features_list'], track_info['area_list']) if area >= median_area]

#                 if valid_features:
#                     # Calculate the mean of the valid feature vectors
#                     mean_features = np.mean(valid_features, axis=0)
#                     features_dict[track_id] = {
#                         'track_id': track_id,
#                         'frame_id': track_info['first_frame'],  # Use the first appearance frame
#                         'class_label': track_info['class_label'],
#                         'features': mean_features.tolist()
#                     }

#     return features_dict


# def trak_and_extract_features_nofilter(model, video_path):
#     class_labels = ["Bicycle", "Bus", "Car", "LCV", "Three-Wheeler", "Truck", "Two-Wheeler"]

#     cap = cv2.VideoCapture(video_path)
#     features_dict = {}
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_id = 0
    
#     # Dictionary to store the features for each track_id
#     running_features = {}

#     with tqdm(total=total_frames, desc="Processing Video") as pbar:
#         while cap.isOpened():
#             # Read the next frame from the video
#             success, frame = cap.read()
#             if success:
#                 frame_id += 1
#                 # Perform tracking on the current frame
#                 results = model.track(frame, persist=True, verbose=False, iou=0.10)
#                 pbar.update(1)

#                 if results[0].boxes:
#                     boxes = results[0].boxes.xywh.cpu().numpy()
#                     track_ids = results[0].boxes.id
#                     classes = results[0].boxes.cls

#                     if track_ids is not None and classes is not None:
#                         track_ids = track_ids.int().cpu().tolist()
#                         classes = classes.int().cpu().tolist()

#                         # Process each detected object in the frame
#                         for box, track_id, cls in zip(boxes, track_ids, classes):
#                             class_label = class_labels[cls]

#                             # Calculate the bounding box dimensions and extract the cropped image
#                             x, y, w, h = [int(coord) for coord in box]
#                             x2, y2 = x + w, y + h  # Bottom-right corner
#                             cropped_image = frame[y:y2, x:x2]

#                             # Extract features from the cropped image
#                             features = extract_features(cropped_image)

#                             # If track_id is new, initialize its entry
#                             if track_id not in running_features:
#                                 running_features[track_id] = {
#                                     'first_frame': frame_id,
#                                     'class_label': class_label,
#                                     'features_list': []
#                                 }
                            
#                             # Append the extracted features to the list for this track_id
#                             running_features[track_id]['features_list'].append(features)
#                             running_features[track_id]['last_frame'] = frame_id

#             else:
#                 print("Completed")
#                 break

#         cap.release()

#     # Compute the mean feature vector for each track_id, only if it has been in more than 40 frames
#     for track_id, track_info in running_features.items():
#         if len(track_info['features_list']) > 0:
#             # Check if the object has been in the frame for more than 40 frames
#             if track_info['last_frame'] - track_info['first_frame'] > 40:
#                 # Calculate the mean of the feature vectors
#                 mean_features = np.mean(track_info['features_list'], axis=0)
#                 features_dict[track_id] = {
#                     'track_id': track_id,
#                     'frame_id': track_info['first_frame'],  # Use the first appearance frame
#                     'class_label': track_info['class_label'],
#                     'features': mean_features.tolist()
#                 }

#     return features_dict

