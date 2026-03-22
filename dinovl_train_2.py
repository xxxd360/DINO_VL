#得到MAP


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import BertTokenizer, BertModel

LOCAL_PATH = r"C:\Users\xxxd\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
model_Ber = BertModel.from_pretrained(LOCAL_PATH, local_files_only=True)
import cv2
import json
from groundeddino_vl import load_model, predict
import supervision as sv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_single_image(image_path, json_path, model_config, model_checkpoint):
    # 加载模型
    model = load_model(config_path=model_config, checkpoint_path=model_checkpoint, device="cuda")

    # 加载COCO数据
    coco_api = COCO(json_path)
    coco_classes = [cat['name'] for cat in coco_api.dataset['categories']]
    name_to_id = {name: idx + 1 for idx, name in enumerate(coco_classes)}

    # 获取图像ID
    img_id = int(os.path.basename(image_path).split('.')[0])

    # 获取真实标注
    gt_ann_ids = coco_api.getAnnIds(imgIds=[img_id])
    gt_annotations = coco_api.loadAnns(gt_ann_ids)
    if not gt_annotations:
        return None

    # 构建提示词
    gt_labels = [coco_classes[ann['category_id'] - 1] for ann in gt_annotations]
    text_prompt = " . ".join(list(set(gt_labels)))

    # 读取图像
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 预测
    result = predict(model=model, image=image_rgb, text_prompt=text_prompt, box_threshold=0.3, text_threshold=0.25)

    if len(result.boxes) == 0:
        return None

    # 坐标转换
    height, width = image_bgr.shape[:2]
    boxes_normalized = result.boxes.cpu().numpy()
    boxes_xyxy = np.zeros_like(boxes_normalized)
    boxes_xyxy[:, 0] = (boxes_normalized[:, 0] - boxes_normalized[:, 2] / 2) * width
    boxes_xyxy[:, 1] = (boxes_normalized[:, 1] - boxes_normalized[:, 3] / 2) * height
    boxes_xyxy[:, 2] = (boxes_normalized[:, 0] + boxes_normalized[:, 2] / 2) * width
    boxes_xyxy[:, 3] = (boxes_normalized[:, 1] + boxes_normalized[:, 3] / 2) * height

    # 获取标签和置信度
    if isinstance(result.labels, list):
        detected_labels = result.labels
    else:
        label_ids = result.labels.cpu().numpy()
        detected_labels = [coco_classes[int(i)] if int(i) < len(coco_classes) else "unknown" for i in label_ids]

    confidences = result.scores.cpu().numpy()

    # 创建COCO格式结果
    coco_results = []
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

        if detected_labels[i] in name_to_id:
            category_id = name_to_id[detected_labels[i]]
            score = float(confidences[i])
            coco_results.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": bbox,
                "score": score
            })

    if not coco_results:
        return None

    # 计算mAP
    pred_coco = coco_api.loadRes(coco_results)
    coco_eval = COCOeval(coco_api, pred_coco, 'bbox')
    coco_eval.params.imgIds = [img_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 可视化
    boxes = []
    labels = []
    scores = []
    for res in coco_results:
        x, y, w, h = res['bbox']
        boxes.append([x, y, x + w, y + h])
        labels.append(coco_classes[res['category_id'] - 1])
        scores.append(res['score'])

    detections = sv.Detections(
        xyxy=np.array(boxes),
        confidence=np.array(scores),
        class_id=np.zeros(len(boxes))
    )

    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    texts = [f"{l}: {s:.2f}" for l, s in zip(labels, scores)]
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=texts)

    # 显示结果
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"mAP@0.5: {coco_eval.stats[1]:.3f}, mAP@0.5:0.95: {coco_eval.stats[0]:.3f}")
    plt.show()

    # 返回mAP结果
    return coco_eval.stats[1], coco_eval.stats[0]


# 使用示例
image_path = "D:\coco\images\\train2017\\000000000772.jpg"
json_path = r'D:\coco\annotations\instances_train2017.json'
model_config = "GroundingDINO_SwinT_OGC.py"
model_checkpoint = "groundingdino_swint_ogc.pth"

map_05, map_05_095 = evaluate_single_image(image_path, json_path, model_config, model_checkpoint)
print(f"mAP@0.5: {map_05:.4f}")
print(f"mAP@0.5:0.95: {map_05_095:.4f}")