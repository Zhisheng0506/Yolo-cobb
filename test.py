import torch
from ultralytics import YOLO

model=YOLO("yolo11n-obb.pt")

results=model.train(data="ultralytics/cfg/datasets/DOTAv1.yaml",epochs=100,batch=8,imgsz=512,workers=4,device=0)