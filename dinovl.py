import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'# 禁用 Hugging Face Hub 检查
from transformers import BertTokenizer, BertModel
LOCAL_PATH = r"C:\Users\xxxd\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH,local_files_only=True)
model_Ber = BertModel.from_pretrained(LOCAL_PATH,local_files_only=True)
from groundeddino_vl import load_model, predict
#图像路径检测
# Load model once (auto-downloads weights if needed)
# model = load_model(
#     config_path="GroundingDINO_SwinT_OGC.py",
#     checkpoint_path="groundingdino_swint_ogc.pth",
#     device="cuda"  # or "cpu"
# )

# # Run detection with text prompt
# result = predict(
#     model=model,
#     image="D:\pytorch.pycharm\datasets\\bvn11\images\\val\\0000023_00000_d_0000008.jpg",
#     text_prompt="car . person . tree",  # Objects separated by " . "
#     box_threshold=0.35,  # Confidence threshold for boxes
#     text_threshold=0.25,  # Confidence threshold for text matching
# )
#
# # Access results
# print(f"Found {len(result)} objects")
# for label, score, box in zip(result.labels, result.scores, result.boxes):
#     print(f"{label}: {score:.2f} at {box}")
#

# 图像阵列检测
import cv2
from groundeddino_vl import load_model, predict

# Load image with OpenCV (BGR format)
image_bgr = cv2.imread("/DINO_VL/assert/zidane.jpg")

# Convert BGR to RGB (GroundedDINO-VL expects RGB)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

model = load_model(
    config_path="models/GroundingDINO_SwinT_OGC.py",
    checkpoint_path="models/groundingdino_swint_ogc.pth",
    device="cuda"  # or "cpu"
)

result = predict(
    model=model,
    image=image_rgb,  # Pass RGB numpy array
    text_prompt="person . dog . bird",
)

print(f"Detected {len(result)} objects: {result.labels}")