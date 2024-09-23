README
Overview
This repository contains Python scripts that facilitate object tracking, vehicle re-identification feature extraction, and similarity calculations in videos. The goal of the project is to track objects, extract relevant features, and compute similarities between tracked objects across different camera views. This README provides a detailed explanation of each of the scripts, their purpose, and their functionality.
### Files
1. trak.py
This script handles the process of tracking objects in a video and extracting features for each vehicle detected in each frame.
Key Features:
    * Object Tracking: Using Phase 1 model, the script tracks objects across frames in a video.
    * Feature Extraction: For each tracked object, relevant features such as bounding boxes and class labels are extracted.

    * Processing per Track:
        * For each track_id, the script computes the mean of relevant feature vectors associated with that track.
        * The bounding box with the median area and the corresponding frame number is also stored (for saving annotations).
        * The most frequent class label for each object (track) is identified, to assign label for that vehicle.

    * Output: A dictionary (features_dict) is created, which stores details such as track_id, frame_id, class_label, feature vector, median bounding box, and the corresponding frame number for each tracked object.

This script interacts with the object detection and tracking model (e.g., YOLO) and requires a video input. 

2. extract_feat.py --
This script is focused on extracting features from the tracked objects.
Key Features:
    * Loading Pre-trained Models: This script uses pre-trained models for feature extraction using a ResNet-50 backbone.
    * Feature Computation: For each object detected in the video, a feature vector is computed based on the pre-trained model.
    * Intermediate Processing: After extracting features for each frame, it organizes and processes them further, possibly aggregating them to form a cohesive representation of the object over time.
   
3. app.py --
This script is the main entry point for running the tracking and feature extraction process.It acts as a wrapper that brings together the functionality of other scripts.
Key Features:

    * Entry point: Likely contains functions to accept user input (such as video paths) and invoke the feature extraction/tracking methods.
    * Modular Organization: This script serve as the main execution point, orchestrating the tracking and feature extraction processes defined in the other scripts.
    * Integration: It imports and combine the functionality of other scripts like trak.py and extract_feat.py, streamlining the process for easier execution.

4. calc_sim.py
This script calculates the similarity between extracted features, from multiple cameras.
Key Features:
    * Similarity Calculation: Uses a similarity metric (cosine similarity) to compare feature vectors extracted from tracked objects.
    * Object Matching Across Cameras:
        * Objects detected from different camera views can be compared to determine if they represent the same real-world object.
        * It computes the best match based on the similarity score.
    * Thresholding and Filtering: A threshold is applied to the similarity score to filter out weak matches. Only matches that exceed the threshold are considered valid.
    * Directional Flow: The script also tracks the order in which an object appears across cameras (e.g., whether it moves from camera A to camera B or vice versa).
This script plays a key role in matching and aligning tracked objects across different videos based on their feature representations.

### Workflow
1. Tracking: The trak.py script processes a video, tracks objects, and extracts bounding boxes and features for each object.
2. Feature Extraction: The extract_feat.py script uses pre-trained deep learning models to compute feature vectors for the tracked objects.
3. Similarity Calculation: The calc_sim.py script compares the feature vectors across different views or frames, identifying potential matches of the same object.
4. Integration: The app.py script serves as the entry point to combine these components, allowing a user to run the full pipeline from video input to feature output and similarity matching.


### Requirements

The following Python libraries are likely required for running the scripts:
* YOU NEED TO DOWNLOAD THE FASTREID WEIGHTS (TRAINED ON VeRI-WILD)
* opencv-python for video processing.
* torch or tensorflow for deep learning model inference.
* tqdm for progress tracking.
* numpy for numerical computations.
* scipy or sklearn for similarity calculations (e.g., cosine similarity).

### How to Run
1. Install Dependencies: Install the necessary dependencies using pip:bashCopy codepip install opencv-python tqdm torch numpy scipy

2. Run the Application: To run the full pipeline, execute app.py with the required input.json and output_directory.
 
3. Output: The program will output a set of feature vectors for each tracked object and a similarity matrix comparing the objects across different videos.

