#拿预训练模型直接推理
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'# 禁用 Hugging Face Hub 检查
from transformers import BertTokenizer, BertModel
LOCAL_PATH = r"C:\Users\xxxd\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH,local_files_only=True)
model_Ber = BertModel.from_pretrained(LOCAL_PATH,local_files_only=True)
import cv2
import json
from groundeddino_vl import load_model, predict
import supervision as sv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# 1. 加载模型 (预训练权重)
model = load_model(
    config_path="models/GroundingDINO_SwinT_OGC.py",
    checkpoint_path="models/groundingdino_swint_ogc.pth",
    device="cuda"
)

json_path = r'D:\coco\annotations\instances_train2017.json'
# 2. 给出类别
def check_category_overlap(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取 COCO 官方所有的类别名称 (80 个)
    coco_categories = [cat['name'] for cat in data['categories']]
    return coco_categories
coco_classes = check_category_overlap(json_path)

# 3. 构造 Prompt (用点号分隔)
text_prompt = " . ".join(coco_classes)

# 4. 读取图片
image_path = "D:\coco\images\\train2017\\000000000772.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 5. 运行预测
result = predict(
    model=model,
    image=image_rgb,
    text_prompt=text_prompt,
    box_threshold=0.45,  # 可以适当调整
    text_threshold=0.35
)

# 6. 可视化 (同类同色)
if len(result.boxes) > 0:
    # 1. 提取边界框 (xyxy 格式) - 需要转换为像素坐标
    if hasattr(result.boxes, 'cpu'):
        boxes_normalized = result.boxes.cpu().numpy()
    else:
        boxes_normalized = result.boxes

    # 转换归一化坐标到像素坐标
    height, width = image_bgr.shape[:2]
    boxes_xyxy_pixel = np.zeros_like(boxes_normalized)
    boxes_xyxy_pixel[:, 0] = (boxes_normalized[:, 0] - boxes_normalized[:, 2] / 2) * width  # x1
    boxes_xyxy_pixel[:, 1] = (boxes_normalized[:, 1] - boxes_normalized[:, 3] / 2) * height  # y1
    boxes_xyxy_pixel[:, 2] = (boxes_normalized[:, 0] + boxes_normalized[:, 2] / 2) * width  # x2
    boxes_xyxy_pixel[:, 3] = (boxes_normalized[:, 1] + boxes_normalized[:, 3] / 2) * height  # y2

    # 2. 提取置信度
    if hasattr(result.scores, 'cpu'):
        confidences = result.scores.cpu().numpy()
    else:
        confidences = result.scores

    # 3. 提取标签 (关键步骤)
    if hasattr(result, 'labels') and result.labels is not None:
        if isinstance(result.labels, list):
            labels = result.labels
        else:
            label_ids = result.labels.cpu().numpy() if hasattr(result.labels, 'cpu') else result.labels
            labels = [coco_classes[int(i)] if i < len(coco_classes) else f"class_{i}" for i in label_ids]

    # 4. 手动创建 supervision Detections 对象
    detections = sv.Detections(
        xyxy=boxes_xyxy_pixel,  # 使用转换后的像素坐标
        confidence=confidences,
        class_id=np.array([coco_classes.index(l) if l in coco_classes else -1 for l in labels]),
    )

    # 修正 class_id 以确保连续
    unique_labels = sorted(list(set(labels)))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    detections.class_id = np.array([label_map[l] for l in labels])

    # 5. 绘制结果
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    texts = [f"{l}: {c:.2f}" for l, c in zip(labels, detections.confidence)]
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=texts)

    # 保存图片
    cv2.imwrite("result.jpg", annotated)

    # 使用 matplotlib 显示图片
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Detection Result")
    plt.show()
else:
    print("⚠️ 未检测到任何目标。")

    # 显示原图
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("No Detection Found")
    plt.show()