#批量处理
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'# 禁用 Hugging Face Hub 检查
from transformers import BertTokenizer, BertModel
LOCAL_PATH = r"C:\Users\xxxd\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH,local_files_only=True)
model_Ber = BertModel.from_pretrained(LOCAL_PATH,local_files_only=True)
from pathlib import Path
from groundeddino_vl import load_model, predict

# Load model once
model = load_model(
    config_path="models/GroundingDINO_SwinT_OGC.py",
    checkpoint_path="models/groundingdino_swint_ogc.pth",
    device="cuda"  # or "cpu"
)
# Get all images in directory
image_dir = Path("images/")
image_paths = list(image_dir.glob("*.jpg"))

# Process each image
results = []
for image_path in image_paths:
    result = predict(
        model=model,
        image=str(image_path),
        text_prompt=" gloves. earphones . cup",
        box_threshold=0.35,
    )
    results.append({
        "image": image_path.name,
        "detections": len(result),
        "labels": result.labels,
        "scores": result.scores,
    })
#
# # Save results to JSON
# import json
# with open("results1.json", "w") as f:
#     json.dump(results, f, indent=2, default=str)

