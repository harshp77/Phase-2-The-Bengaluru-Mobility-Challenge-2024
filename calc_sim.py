import json
import numpy as np
import pathlib
import cv2

def calculate_similarity(featuresA, featuresB):
    # Convert lists to numpy arrays
    featuresA = np.array(featuresA)
    featuresB = np.array(featuresB)
    
    # Calculate cosine similarity
    epsilon = 1e-10  # To prevent division by zero
    dot_product = np.dot(featuresA, featuresB)
    normA = np.linalg.norm(featuresA)
    normB = np.linalg.norm(featuresB)
    
    similarity = dot_product / (normA * normB + epsilon)  # Adding epsilon to prevent division by zero
    
    return similarity

def create_camera_matrix(data, num_cameras, output_dir, video_paths):
    class_names = set(["Bicycle", "Bus", "Car", "LCV", "Three-Wheeler", "Truck", "Two-Wheeler"])
    matrices = {class_name: np.zeros((num_cameras, num_cameras)) for class_name in class_names}
    
    uid_matched = {}
    vehicle_id_counter = 0

    # Open video capture for each camera
    caps = {cam_name: cv2.VideoCapture(video_paths[cam_name]) for cam_name in data}
    
    for camA in data:
        uid_matched = {camB: set() for camB in data if camB != camA}

        for trackA_id, trackA in data[camA].items():
            class_name = trackA['class_label']
            uidA = trackA['track_id']
            best_match_score = -1
            best_camB = None
            best_uidB = None
            
            for camB in data:
                if camA == camB:
                    continue
                
                for trackB_id, trackB in data[camB].items():
                    uidB = trackB['track_id']
                    
                    if trackA['class_label'] != trackB['class_label']:
                        continue
                    
                    if uidB in uid_matched[camB]:
                        continue
                    
                    similarity_score = calculate_similarity(trackA['features'], trackB['features'])
                    
                    if similarity_score > best_match_score:
                        best_match_score = similarity_score
                        best_camB = camB
                        best_uidB = uidB
                        best_track_b = trackB
            
            if best_camB and best_match_score > 0.72:
                camA_idx = int(camA[-1]) - 1
                camB_idx = int(best_camB[-1]) - 1
                matrices[class_name][camA_idx][camB_idx] += 1
                
                uid_matched[best_camB].add(best_uidB)
                
                vehicle_id_counter += 1
                vehicle_id = f"Vehicle_{vehicle_id_counter}"
                
                # Get median frame and bounding box for both cameras
                frameA = trackA['median_frame']
                bboxA = trackA['median_bbox']
                
                frameB = best_track_b['median_frame']
                bboxB = best_track_b['median_bbox']
                
                # Save annotated frames with correct bounding boxes
                save_annotated_frame(caps[camA], frameA, bboxA, camA, class_name, vehicle_id, output_dir)
                save_annotated_frame(caps[best_camB], frameB, bboxB, best_camB, class_name, vehicle_id, output_dir)

    for cap in caps.values():
        cap.release()
    
    return matrices

def save_annotated_frame(cap, frame_number, bbox, cam_name, class_name, vehicle_id, output_dir):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    if not success:
        print(f"Could not retrieve frame {frame_number} from {cam_name}")
        return
    
    x, y, w, h = bbox
    
    # Draw bounding box and label on the frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"{class_name}_{cam_name}_{frame_number}_{vehicle_id}"
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Create output directory if it doesn't exist
    pathlib.Path(f"{output_dir}/Images/{class_name}/").mkdir(parents=True, exist_ok=True)
    
    # Save the annotated frame
    output_path = f"{output_dir}/Images/{class_name}/{label}.jpg"
    cv2.imwrite(output_path, frame)

def cal_sim(data, output_dir, video_paths):
    num_cameras = len(data)
    
    matrices = create_camera_matrix(data, num_cameras, output_dir, video_paths)
    
    pathlib.Path(f'{output_dir}/Matrices/').mkdir(parents=True, exist_ok=True)
    
    for class_name, matrix in matrices.items():
        with open(f"{output_dir}/Matrices/{class_name}.json", "w") as f:
            json.dump(matrix.tolist(), f)



#################################### DEAD CODE ##########################################################

# def calculate_similarity(featuresA, featuresB, threshold=0.8):
#     # Compute cosine similarity (or replace with your own logic)
#     similarity = np.dot(featuresA, featuresB) / (np.linalg.norm(featuresA) * np.linalg.norm(featuresB))
#     return similarity

# import numpy as np


# def create_camera_matrix2(data, num_cameras):
#     # Initialize matrices for each class type with zeros
#     class_names = set(["Bicycle", "Bus", "Car", "LCV", "Three-Wheeler", "Truck", "Two-Wheeler"])
#     matrices = {class_name: np.zeros((num_cameras, num_cameras)) for class_name in class_names}
    
#     # Dictionary to track matched UIDs for each camB (prevents reuse of UIDs from camB for same camA)
#     uid_matched = {}
    
#     # Iterate through all cameras (camera A)
#     for camA in data:
#         # Refresh the uid_matched dictionary for a new camA
#         uid_matched = {camB: set() for camB in data if camB != camA}
        
#         # For each track in camera A
#         for trackA_id, trackA in data[camA].items():
#             class_name = trackA['class_label']
#             uidA = trackA['track_id']
#             frameA = trackA['frame_id']  # First appearance frame in camera A
#             best_match_score = -1  # To keep track of the highest similarity
#             best_camB = None
#             best_uidB = None
#             best_frameB = None
            
#             # Now loop through all other cameras (camera B) for comparison
#             for camB in data:
#                 if camA == camB:
#                     continue  # Skip comparing the same camera
                
#                 # Compare with each track in camera B
#                 for trackB_id, trackB in data[camB].items():
#                     uidB = trackB['track_id']
#                     frameB = trackB['frame_id']  # First appearance frame in camera B
                    
#                     # Skip if the class label doesn't match
#                     if trackA['class_label'] != trackB['class_label']:
#                         continue
                    
#                     # Skip if uidB from camB is already matched with some uid from camA
#                     if uidB in uid_matched[camB]:
#                         continue
                    
#                     # Calculate similarity
#                     similarity_score = calculate_similarity(trackA['features'], trackB['features'])
                    
#                     # Choose the best match based on similarity
#                     if similarity_score > best_match_score:
#                         best_match_score = similarity_score
#                         best_camB = camB
#                         best_uidB = uidB
#                         best_frameB = frameB
            
#             # If a best match is found and meets a threshold, update the matrices and mark uidB as matched
#             if best_camB and best_match_score > 0.851:  # You can adjust the threshold as needed
#                 camA_idx = int(camA[-1]) - 1  # Assuming camA is in the format "Cam1", "Cam2", etc.
#                 camB_idx = int(best_camB[-1]) - 1
                
#                 # Compare frame_id values to determine direction
#                 if frameA < best_frameB:
#                     # If the object first appeared in camA, it's A -> B
#                     matrices[class_name][camA_idx][camB_idx] += 1
#                 else:
#                     # If the object first appeared in camB, it's B -> A
#                     matrices[class_name][camB_idx][camA_idx] += 1
                
#                 # Mark this uidB as matched so it's not used again for this cam.A
#                 uid_matched[best_camB].add(best_uidB)
    
#     return matrices

