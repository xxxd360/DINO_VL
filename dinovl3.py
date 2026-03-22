import json

# 加载训练集标注
with open(r'D:\coco\annotations\instances_train2017.json', 'r') as f:
    data = json.load(f)

# 获取所有类别信息
categories = data['categories']

print(f"COCO 数据集共有 {len(categories)} 个类别：")
for cat in categories:
    print(f"ID: {cat['id']:2d}, 名称: {cat['name']}")
