import json
import os

# ================= 配置区域 =================
# 输入文件路径 (你下载的原始标注文件)
input_json_path = r'D:\coco\annotations\instances_train2017.json'

# 输出文件路径 (过滤后只包含 Seen 类的文件)
output_json_path = r'D:\coco\annotations\instances_train2017_seen_only.json'

# 标准的 15 个 Unseen 类别 (基于 ECCV 2018 Zero-Shot Split-1)
# 注意：COCO 类别名称必须完全匹配，包括空格和单复数
unseen_classes_list = [
'bench',
'bicycle',
'bus',
'cow',
'frisbee',
'handbag',
'hot dog',
'keyboard',
'microwave',
'oven',
'suitcase',
'toaster',
'traffic light',
'train',
'umbrella',
]


# ===========================================

def filter_unseen_annotations(input_path, output_path, unseen_names):
    print(f"📂 正在读取文件: {input_path} ...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 建立类别名称到 ID 的映射，并找出哪些 ID 是 Unseen 的
    categories = data['categories']
    all_cat_ids = {cat['id']: cat['name'] for cat in categories}

    unseen_ids = set()
    seen_categories = []

    for cat in categories:
        if cat['name'] in unseen_names:
            unseen_ids.add(cat['id'])
            print(f"   ❌ 剔除类别 (Unseen): {cat['name']} (ID: {cat['id']})")
        else:
            seen_categories.append(cat)

    print(f"\n✅ 共识别出 {len(unseen_ids)} 个 Unseen 类别，{len(seen_categories)} 个 Seen 类别。")

    # 2. 过滤 Annotations
    # 只保留 category_id 不在 unseen_ids 集合中的标注框
    original_ann_count = len(data['annotations'])
    filtered_annotations = [
        ann for ann in data['annotations']
        if ann['category_id'] not in unseen_ids
    ]
    filtered_ann_count = len(filtered_annotations)

    removed_count = original_ann_count - filtered_ann_count
    print(f"🗑️  原始标注框数量: {original_ann_count}")
    print(f"🗑️  剔除的标注框数量: {removed_count} (属于 Unseen 类)")
    print(f"✅ 保留的标注框数量: {filtered_ann_count} (属于 Seen 类)")

    # 3. 构建新的 JSON 数据结构
    new_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": data["images"],  # 图片信息全部保留 (即使图里有 Unseen 物体，我们只是不看它的框)
        "annotations": filtered_annotations,  # 只保留 Seen 类的框
        "categories": seen_categories  # 只保留 Seen 类的类别定义
    }

    # 4. 保存到新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, separators=(',', ':'))



if __name__ == "__main__":

    filter_unseen_annotations(input_json_path, output_json_path, unseen_classes_list)