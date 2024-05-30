import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET

def find_filename(folder_path:str=None,     # 資料夾位置
                  File_extension:str=None): # 副檔名
    """  尋找資料夾內 '特定副檔名' 的 '所有檔案的檔名'    """
    filesname = [file for file in os.listdir(folder_path) if file.endswith(f'.{File_extension}')]
    return filesname,len(filesname)



class ContainerDataset(Dataset):
    def __init__(self, image_dir, annot_dir, transform=None):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        annot_path = os.path.join(self.annot_dir, self.images[idx].replace('.jpg', '.xml'))

        image = Image.open(image_path).convert("RGB")
        tree = ET.parse(annot_path)
        root = tree.getroot()
        
        boxes = []
        for member in root.findall('object'):
            xmin = int(member.find('bndbox/xmin').text)
            ymin = int(member.find('bndbox/ymin').text)
            xmax = int(member.find('bndbox/xmax').text)
            ymax = int(member.find('bndbox/ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        if self.transform:
            image, boxes = self.transform(image, boxes)

        return image, boxes
class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor()
        ])
    
    def __call__(self, image, boxes):
        image = self.transform(image)
        # 如果你需要對標註框也做相應的轉換（例如，縮放標註框），你可以在這裡添加
        # 現在假設我們不需要對標註框做轉換，直接返回即可
        return image, boxes
    

def predictions_to_coco_json(results):
    image_id = 0
    annotation_id = 0
    coco_results = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "container"}]  # 確保這個類別 ID 和你的模型預測匹配
    }

    for result in results:
        image_info = {
            "id": image_id,
            "file_name": result.path.split('/')[-1],
            "width": result.orig_shape[1],
            "height": result.orig_shape[0]
        }
        coco_results["images"].append(image_info)

        boxes = result.boxes.xyxy.cpu().numpy()  # 確保資料是在 CPU 上並轉為 numpy 便於處理
        scores = result.boxes.conf.cpu().numpy()  # 同上
        classes = result.boxes.cls.cpu().numpy()  # 同上

        for box, score, class_id in zip(boxes, scores, classes):
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_id) + 1,  # COCO 的類別 ID 通常從 1 開始
                "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                "score": float(score),
                "iscrowd": 0,
                "area": (box[2] - box[0]) * (box[3] - box[1])
            }
            coco_results["annotations"].append(annotation)
            annotation_id += 1

        image_id += 1

    return coco_results
def predictions_to_coco_json_ssd(results, image_paths):
    coco_results = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 添加一個類別（假設只有一個類別，ID 為 1，名稱為 "container"）
    coco_results["categories"].append({
        "id": 1,
        "name": "container"
    })

    annotation_id = 1

    for image_id, (result, image_path) in enumerate(zip(results, image_paths)):
        image_info = {
            "id": image_id,
            "file_name": image_path
        }
        coco_results["images"].append(image_info)

        for box, score, label in zip(result[0]['boxes'], result[0]['scores'], result[0]['labels']):
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(label.item()),  # 假設 label 是 container 類別，ID 為 1
                "bbox": [box[0].item(), box[1].item(), (box[2] - box[0]).item(), (box[3] - box[1]).item()],
                "score": score.item(),
                "iscrowd": 0,
                "area": (box[2] - box[0]).item() * (box[3] - box[1]).item()
            }
            coco_results["annotations"].append(annotation)
            annotation_id += 1

    return coco_results


def xmls_to_coco_json(xml_folder):
    image_id = 0
    annotation_id = 0
    coco_results = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "container"}]
    }

    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_folder, xml_file))
            root = tree.getroot()
            image_info = {
                "id": image_id,
                "file_name": xml_file.replace('.xml', '.jpg'),
                "width": int(root.find('size/width').text),
                "height": int(root.find('size/height').text)
            }
            coco_results["images"].append(image_info)

            for member in root.findall('object'):
                bndbox = member.find('bndbox')
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                             int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text),
                             int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)],
                    "iscrowd": 0,
                    "area": (int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text)) *
                            (int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text))
                }
                coco_results["annotations"].append(annotation)
                annotation_id += 1

            image_id += 1

    return coco_results



from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def compute_metrics(predictions, ground_truths):
    coco_gt = COCO()
    coco_gt.dataset = ground_truths
    coco_gt.createIndex()

    coco_dt = COCO()
    coco_dt.dataset = predictions
    coco_dt.createIndex()

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "mAP50": stats[1],
        "mAP50-95": np.mean(stats[0:10]),
        "Precision": stats[0],
        "Recall": stats[8],
        "F1-Score": 2 * (stats[0] * stats[8]) / (stats[0] + stats[8]) if (stats[0] + stats[8]) > 0 else 0
    }


from torchvision.io import read_image
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import convert_image_dtype

def load_images(file_list):
    images = []
    for file_path in file_list:
        img = read_image(file_path)
        img = convert_image_dtype(img, dtype=torch.float32)  # 將圖片轉換為浮點類型
        images.append(img)
    return images

def predict_images(model, images, device='cpu'):
    model.to(device)
    model.eval()
    all_predictions = []
    
    for img in images:
        img = img.to(device)
        with torch.no_grad():
            predictions = model([img])
        all_predictions.append(predictions)
    
    return all_predictions
# def load_images(file_list):
#     # 讀取所有圖片並轉換為張量
#     images = [read_image(file_path) for file_path in file_list]
    
#     # 將圖片堆疊成一個批次
#     # batch = torch.stack(images)
    
#     return np.array(images)