import json
import os

# ================= 配置区域 =================
# 你的原始标注文件路径
json_path = r'D:\coco\annotations\instances_train2017.json'

my_unseen_list = [
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

def check_category_overlap(json_file, user_list):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取 COCO 官方所有的类别名称 (80 个)
    coco_categories = [cat['name'] for cat in data['categories']]
    coco_set = set(coco_categories)

    user_set = set(user_list)


    # 1. 找出完全匹配的 (成功剔除的)
    matched = user_set.intersection(coco_set)

    # 2. 找出你写了但 COCO 里没有的 (拼写错误)
    not_in_coco = user_set - coco_set

    # 3. 找出 COCO 里有但你没写的 (漏掉的)
    unmatched_in_user = user_set - matched

    if not_in_coco:
        print("\n❌(你的列表里有，但 COCO 里找不到):")
        for name in sorted(list(not_in_coco)):
            print(f"    '{name}'  ")

    if unmatched_in_user:
        # 这通常就是 not_in_coco 的部分
        pass


if __name__ == "__main__":
    if not os.path.exists(json_path):
        print(f"❌ 文件不存在: {json_path}")
    else:
        check_category_overlap(json_path, my_unseen_list)