#可视化
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'# 禁用 Hugging Face Hub 检查
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from transformers import BertTokenizer, BertModel
LOCAL_PATH = r"C:\Users\xxxd\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH,local_files_only=True)
model_Ber = BertModel.from_pretrained(LOCAL_PATH,local_files_only=True)
# from groundeddino_vl import load_model, predict, annotate
# import cv2
#
# model = load_model(
#     config_path="GroundingDINO_SwinT_OGC.py",
#     checkpoint_path="groundingdino_swint_ogc.pth",
#     device="cuda"  # or "cpu"
# )
# # Load image (returns RGB numpy array)
# image_rgb = cv2.cvtColor(cv2.imread("D:\pytorch.pycharm\zidane.jpg"), cv2.COLOR_BGR2RGB)
#
# # Run detection
# result = predict(
#     model=model,
#     image=image_rgb,
#     text_prompt="person . car . bicycle",
# )
#
# # Annotate image (returns BGR format for OpenCV)
# annotated = annotate(
#     image=image_rgb,
#     result=result,
#     show_labels=True,
#     show_confidence=True,
# )
#
# # Save or display
# cv2.imwrite("output.jpg", annotated)
# cv2.imshow("Detection Result", annotated)
# cv2.waitKey(0)
#


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from groundeddino_vl import load_model, predict

model = load_model(
    config_path="models/GroundingDINO_SwinT_OGC.py",
    checkpoint_path="models/groundingdino_swint_ogc.pth",
    device="cuda"  # or "cpu"
)
result = predict(model, "/DINO_VL/assert/zidane.jpg", "person . dog")

# Get denormalized boxes in xyxy format
boxes = result.to_xyxy(denormalize=True)

# Plot with matplotlib
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(plt.imread("/DINO_VL/assert/zidane.jpg"))

# Draw bounding boxes
for box, label, score in zip(boxes, result.labels, result.scores):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1

    # Draw rectangle
    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

    # Add label
    ax.text(
        x1, y1 - 5,
        f"{label}: {score:.2f}",
        color='white',
        fontsize=12,
        bbox=dict(facecolor='red', alpha=0.7)
    )

plt.axis('off')
plt.tight_layout()
plt.savefig("annotated_output.jpg", dpi=150, bbox_inches='tight')
plt.show()