#带监督的低级API配合 Supervision 库
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'# 禁用 Hugging Face Hub 检查
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from transformers import BertTokenizer, BertModel
LOCAL_PATH = r"C:\Users\xxxd\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH,local_files_only=True)
model_Ber = BertModel.from_pretrained(LOCAL_PATH,local_files_only=True)
import cv2
import numpy as np
from groundeddino_vl.utils.inference import Model
import supervision as sv

# Load image (BGR format)
image_bgr = cv2.imread("/DINO_VL/assert/zidane.jpg")

# Initialize model with low-level API
model = Model(
    model_config_path="models/GroundingDINO_SwinT_OGC.py",
    model_checkpoint_path="models/groundingdino_swint_ogc.pth",
    device="cuda"
)

# Predict with caption (returns sv.Detections + labels)
detections, labels = model.predict_with_caption(
    image=image_bgr,
    caption="person . car . bicycle",
    box_threshold=0.35,
    text_threshold=0.25,
)
# 1. 确保 detections 有 class_id 属性 (有些旧版本 supervision 需要手动初始化)
if not hasattr(detections, 'class_id') or detections.class_id is None:
    # 2. 获取所有唯一的标签名称，并排序 (保证每次运行颜色分配一致)
    unique_labels = sorted(list(set(labels)))

    # 3. 创建映射字典：标签 -> 数字ID (例如: {'car': 0, 'person': 1})
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # 4. 将每个检测框的标签转换为对应的 ID
    class_ids = np.array([label_map[label] for label in labels])

    # 5. 赋值给 detections 对象
    detections.class_id = class_ids


# Visualize with Supervision
box_annotator = sv.BoxAnnotator()
annotated = box_annotator.annotate(scene=image_bgr, detections=detections)

# Add labels with confidence
label_annotator = sv.LabelAnnotator()
labels_with_conf = [
    f"{label} {conf:.2f}"
    for label, conf in zip(labels, detections.confidence)
]
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels_with_conf
)

cv2.imshow("Result", annotated)
cv2.waitKey(0)