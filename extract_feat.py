import cv2
import torch
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Load FastReID
cfg = get_cfg()
cfg.merge_from_file("Asset/Yamls/re_used/reid.yml")  # Vehicle reid config
model_reid = build_model(cfg)
Checkpointer(model_reid).load("Asset/veri_sbs_R50-ibn.pth")
model_reid.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_reid.to(device)

transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

import numpy as np

def extract_features(image):
    """
    Extract features from an image using the FastReID model and reduce dimensionality.
    
    Args:
        image (numpy array): Input image.
    Returns:
        list: Reduced features as a list of floats.
    """
    # Convert the image to PIL and then tensor
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Extract features using the model
    with torch.no_grad():
        features = model_reid(image_tensor)  # Assume the output is a 1D feature vector


    return features.cpu().numpy().flatten()






