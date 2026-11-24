import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor
from transformers import BertTokenizer, VisualBertModel, VisualBertForPreTraining

import sys, os

if len(sys.argv) == 1:
    filename:str = "tennis_racket_1.jpg"
else:
    filename:str = sys.argv[1]
    if not os.path.isfile(filename):
        print("Invalid File")
        exit(1)

if len(sys.argv) < 3:
    text:str = "A person hits a ball with a racket"
else:
    text:str = sys.argv[2]

if len(sys.argv) < 4:
    masked_text:str = "A person hits a ball with a [MASK]"
else:
    masked_text:str = sys.argv[3]


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

# 4. Appended: Sentence-Image Match Probability via VisualBertForPreTraining
pretrain_model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre").to(device).eval()

with torch.no_grad():
    pretrain_outputs = pretrain_model(
        **inputs,
        visual_embeds=visual_embeds,
        visual_attention_mask=visual_attention_mask,
        visual_token_type_ids=visual_token_type_ids,
    )

# seq_relationship_logits has shape [1, 2]; softmax turns those 2 logits into probabilities summing to 1.0
match_probs = torch.softmax(pretrain_outputs.seq_relationship_logits, dim=-1)[0]
print("Statement:", text)
print(f"Match probability (index 0): {match_probs[0].item():.4f} ({match_probs[0].item() * 100:.1f}%)")
print(f"Non-match probability (index 1): {match_probs[1].item():.4f} ({match_probs[1].item() * 100:.1f}%)")


# --- Option A: Masked Word Prediction (shows off VisualBERT's strongest pretraining head) ---
masked_inputs = tokenizer(masked_text, return_tensors="pt").to(device)
mask_token_index = (masked_inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

with torch.no_grad():
    mlm_logits = pretrain_model(
        **masked_inputs,
        visual_embeds=visual_embeds,
        visual_attention_mask=visual_attention_mask,
        visual_token_type_ids=visual_token_type_ids,
    ).prediction_logits

top_5_tokens = torch.topk(mlm_logits[0, mask_token_index], 5, dim=-1).indices[0]
print("Masked prompt:", masked_text)
print("Top 5 predictions:", [tokenizer.decode([t]).strip() for t in top_5_tokens])