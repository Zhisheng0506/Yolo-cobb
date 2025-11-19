import torch
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/11/yolo11-obb.yaml')
model.model.to('cpu')
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    y = model.model(x)
print('forward outputs:', y.shape)

print('forward outputs type:', type(y))