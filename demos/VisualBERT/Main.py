import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor
from transformers import BertTokenizer, VisualBertModel

import sys, os

if len(sys.argv) == 1:
    filename:str = "tennis_racket_1.jpg"
else:
    filename:str = sys.argv[1]
    if not os.path.isfile(filename):
        print("Invalid File")
        exit(1)

device:str = "cuda" if torch.cuda.is_available() else "cpu"

""" 1. Load Pretrained Object Detector (Faster R-CNN) """
# Already pre-installed with torchvision
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
detector = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()

# Hook into the ROI box head (before final class/box layers) to grab 1024-d region features,
# then pad/project to 2048-d to match VisualBERT's expected visual_embedding_dim

# Essentially gets the feature vectors before they are turned into boxes
region_features:list = [] # stores the captured tensors
def hook_fn(module, input, output):
    region_features.append(output.detach()) # keeps only the final tensors with .detach()

hook = detector.roi_heads.box_head.register_forward_hook(hook_fn) # attaches the hook_fn function to the detector, returning a hook handler to clean up after

# 2. Load Image & Extract Top Region Features
img = Image.open(filename).convert("RGB")
img_tensor = to_tensor(img).to(device)

# no gradient running. 
with torch.no_grad():
    detections = detector([img_tensor])[0] # use the RCNN and get the tensors

hook.remove() # clean up

# Take top 10 detected regions and match VisualBERT's 2048-dim input
top_k = min(10, region_features[0].shape[0])
raw_feats = region_features[0][:top_k]  # [top_k, 1024]
visual_embeds = torch.nn.functional.pad(raw_feats, (0, 2048 - raw_feats.shape[1])).unsqueeze(0) # pad the 1024 to 2048 with zeros

# 3. Load VisualBertModel & Fuse Text + Image Regions
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased") # downloads and uses google's BERT tokenizer
model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre").to(device).eval()

text = "A person hits a ball with a racket"
inputs = tokenizer(text, return_tensors="pt").to(device)

visual_attention_mask = torch.ones((1, top_k), dtype=torch.long, device=device)
visual_token_type_ids = torch.ones((1, top_k), dtype=torch.long, device=device)

with torch.no_grad():
    outputs = model(
        **inputs,
        visual_embeds=visual_embeds,
        visual_attention_mask=visual_attention_mask,
        visual_token_type_ids=visual_token_type_ids,
    )

print("Detected boxes (top 3):", detections["boxes"][:3].cpu().numpy())
print("Fused sequence shape (text tokens + image regions):", outputs.last_hidden_state.shape)
print("Pooled multimodal representation shape:", outputs.pooler_output.shape)