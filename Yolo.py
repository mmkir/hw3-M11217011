import torch,argparse
from ultralytics import YOLO
from torch.utils.data import DataLoader
from package.function import *


# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    # 訓練模型
    model.train(
        data="D:/學業/碩士課程/機器學習/分組作業/專題三/data.yaml",  # 訓練數據的配置文件
        epochs=200,  # 訓練的輪數
        imgsz=416,  # 圖像大小
        batch=32,  # 批次大小
        device=0,  # 使用的設備（0表示使用第一塊GPU）
        save=True,  # 是否保存模型
        cache=True,  # 是否緩存數據集
        optimizer='Adam',  # 使用的優化器
        patience=20  # 提前停止的耐心值
    )
