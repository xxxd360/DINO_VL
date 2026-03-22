# 多提示集成 (Prompt Ensembling)
import os
import json as json_lib

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import BertTokenizer, BertModel

LOCAL_PATH = r"C:\Users\xxxd\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
model_Ber = BertModel.from_pretrained(LOCAL_PATH, local_files_only=True)
import cv2
from groundeddino_vl import load_model, predict
import supervision as sv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_prompt_ensembles(category_name):
    ensembles = {
        'person': ['person', 'a person', 'human'],
        'cow': ['cow', 'a cow', 'cattle'],
        'car': ['car', 'a car', 'vehicle'],
        'dog': ['dog', 'a dog', 'canine'],
        'default': lambda name: [name, f"a {name}"]
    }
    return ensembles.get(category_name, ensembles['default'](category_name))


def calculate_map_single(coco_api, img_id, detections_list, name_to_id):
    """辅助函数：计算给定检测列表的 mAP"""
    if not detections_list:
        return 0.0, 0.0, []

    coco_results = []
    for det in detections_list:
        x1, y1, x2, y2 = det['bbox']
        coco_results.append({
            "image_id": img_id,
            "category_id": name_to_id[det['label']],
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(det['score'])
        })

    if not coco_results:
        return 0.0, 0.0, []

    try:
        pred_coco = coco_api.loadRes(coco_results)
        coco_eval = COCOeval(coco_api, pred_coco, 'bbox')
        coco_eval.params.imgIds = [img_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[1], coco_eval.stats[0], coco_results
    except Exception as e:
        print(f"Map calculation error: {e}")
        return 0.0, 0.0, []


def evaluate_single_image_ensemble(image_path, json_path, model_config, model_checkpoint, output_json_path=None):
    # 1. 初始化
    model = load_model(config_path=model_config, checkpoint_path=model_checkpoint, device="cuda")
    coco_api = COCO(json_path)
    coco_classes = [cat['name'] for cat in coco_api.dataset['categories']]
    name_to_id = {name: idx + 1 for idx, name in enumerate(coco_classes)}

    img_id = int(os.path.basename(image_path).split('.')[0])
    gt_ann_ids = coco_api.getAnnIds(imgIds=[img_id])
    gt_annotations = coco_api.loadAnns(gt_ann_ids)
    if not gt_annotations:
        print("No ground truth annotations found.")
        return None

    gt_labels = list(set([coco_classes[ann['category_id'] - 1] for ann in gt_annotations]))
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_bgr.shape[:2]

    all_detections = []
    prompt_results = []

    # 2. 多提示词推理
    for label in gt_labels:
        prompts = get_prompt_ensembles(label)
        for prompt_text in prompts:
            try:
                result = predict(model=model, image=image_rgb, text_prompt=prompt_text, box_threshold=0.25,
                                 text_threshold=0.20)
                current_dets = []

                if len(result.boxes) > 0:
                    boxes_norm = result.boxes.cpu().numpy()
                    boxes_xyxy = np.zeros_like(boxes_norm)
                    boxes_xyxy[:, 0] = (boxes_norm[:, 0] - boxes_norm[:, 2] / 2) * width
                    boxes_xyxy[:, 1] = (boxes_norm[:, 1] - boxes_norm[:, 3] / 2) * height
                    boxes_xyxy[:, 2] = (boxes_norm[:, 0] + boxes_norm[:, 2] / 2) * width
                    boxes_xyxy[:, 3] = (boxes_norm[:, 1] + boxes_norm[:, 3] / 2) * height

                    scores = result.scores.cpu().numpy()
                    for i in range(len(boxes_xyxy)):
                        det = {'bbox': boxes_xyxy[i], 'score': scores[i], 'label': label}
                        current_dets.append(det)
                        all_detections.append(det)

                map_05, map_05_95, _ = calculate_map_single(coco_api, img_id, current_dets, name_to_id)
                prompt_results.append({
                    'prompt': f"{label}: '{prompt_text}'",
                    'count': len(current_dets),
                    'mAP@0.5': map_05,
                    'mAP@0.5:0.95': map_05_95
                })
            except Exception as e:
                continue

    if not all_detections:
        print("No detections found.")
        return None

    # 3. NMS 融合
    final_detections = []
    unique_labels = list(set([d['label'] for d in all_detections]))

    for lbl in unique_labels:
        items = [d for d in all_detections if d['label'] == lbl]
        c_boxes = np.array([d['bbox'] for d in items])
        c_scores = np.array([d['score'] for d in items])

        temp_dets = sv.Detections(xyxy=c_boxes, confidence=c_scores)
        nms_dets = temp_dets.with_nms(threshold=0.5, class_agnostic=True)

        for i in range(len(nms_dets.xyxy)):
            final_detections.append({
                'bbox': nms_dets.xyxy[i],
                'score': nms_dets.confidence[i],
                'label': lbl,
                'category_id': name_to_id[lbl]
            })

    # 4. 计算最终 mAP 并准备 JSON 数据
    ens_map_05, ens_map_05_95, final_coco_results = calculate_map_single(coco_api, img_id, final_detections, name_to_id)

    # 5. 打印对比表格
    print(f"\n{'Prompt':<25} | {'Count':<5} | {'mAP@0.5':<8} | {'mAP@0.5:0.95':<12}")
    print("-" * 55)
    for res in prompt_results:
        print(f"{res['prompt']:<25} | {res['count']:<5} | {res['mAP@0.5']:.4f}   | {res['mAP@0.5:0.95']:.4f}")
    print("-" * 55)
    print(f"{'ENSEMBLE (Fused)':<25} | {len(final_detections):<5} | {ens_map_05:.4f}   | {ens_map_05_95:.4f}")
    print("-" * 55)

    # 6. 保存 JSON 结果
    if output_json_path is None:
        output_json_path = f"detection_results_img{img_id}.json"

    try:
        with open(output_json_path, 'w') as f:
            json_lib.dump(final_coco_results, f, indent=2)
        print(f"   包含 {len(final_coco_results)} 个检测框")
    except Exception as e:
        print(f"保存 JSON 失败: {e}")

    # 7. 可视化
    if final_detections:
        vis_boxes = [d['bbox'] for d in final_detections]
        vis_labels = [d['label'] for d in final_detections]
        vis_scores = [d['score'] for d in final_detections]

        detections_vis = sv.Detections(xyxy=np.array(vis_boxes), confidence=np.array(vis_scores),
                                       class_id=np.zeros(len(vis_boxes)))
        annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(scene=image_bgr.copy(),
                                                                                detections=detections_vis)
        texts = [f"{l}: {s:.2f}" for l, s in zip(vis_labels, vis_scores)]
        annotated = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(scene=annotated,
                                                                                  detections=detections_vis,
                                                                                  labels=texts)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Ensemble Result\nmAP@0.5: {ens_map_05:.3f}, mAP@0.5:0.95: {ens_map_05_95:.3f}")
        plt.show()

    return ens_map_05, ens_map_05_95


if __name__ == "__main__":
    image_path = "D:\coco\images\\train2017\\000000000772.jpg"
    json_path = r'D:\coco\annotations\instances_train2017.json'
    model_config = "GroundingDINO_SwinT_OGC.py"
    model_checkpoint = "groundingdino_swint_ogc.pth"


    output_file = "results/results2.json"


    evaluate_single_image_ensemble(image_path, json_path, model_config, model_checkpoint, output_json_path=output_file)