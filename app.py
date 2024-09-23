import json
import pathlib
import sys
from ultralytics import YOLO
from trak import trak_and_extract_features
from calc_sim import cal_sim

input_file = sys.argv[1]
output_dir = sys.argv[2]
output_dir = output_dir.split('app/')[-1]
BASE_DIR = str(pathlib.Path().resolve())
with open(input_file) as f:
    video_paths = json.load(f)
model = YOLO('Asset/640.pt')

tracking = {cams:[] for cams in video_paths.keys()}
feat = {cams:[] for cams in video_paths.keys()}

# Returns FEATURE MEAN AFTER TRACKING AND FILTERING
for ele in video_paths:
    feat[ele] = trak_and_extract_features(model, video_paths[ele])   

# CALCULATE THE SIMILARITY AND SAVES THE OUTPUT
cal_sim(feat, output_dir, video_paths)
