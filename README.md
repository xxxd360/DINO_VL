DINO  
---  
## 项目结构  
├─assert ----图片资源  
├─config ----配置文件  
├─result ----运行结果  
├─models ----模型权重  
├─scripts ----辅助脚本  
---  
## 依赖安装  
COCO数据集下载位置 ：https://app.roboflow.com/ds/qHK8Q42lc8?key=27IHm68o5o  
DINO预训练权重下载位置  ：https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
---
## 项目解析  
文本引导的零样本检测模型  
复现路线：DINO（groundeddino_vl）   
数据集（MSCOCO)  
选择的优化点：多提示集成 (Prompt Ensembling)  
Train1：DINO使用生成检测框  
Train2：推理与MAP  
result1对应的是批量处理生成的JSON文件：数据用的是我的自定义文件  
result2与Prompt Ensembling_result.txt对应的是Train3做的多提示集成结果  
---
## DINO的使用  
1.先下载好两个预训练的配置文件      
2.下载COCO数据集，完成65——15分类     
2.构建好prompt导入模型预测    
3.NMS融合后用pycocotools获取MAP    
4.比较在不同提示词的效果  
---
## DINO原理学习  
### 师生网络 + 对比学习 + 动量更新   
DINO 抛弃负样本，仅通过师生网络的蒸馏，让模型学会对同一图像的不同视角输出一致的特征，同时对不同图像输出差异化特征  
与YOLO不同的是DINO是自监督预训练框架
