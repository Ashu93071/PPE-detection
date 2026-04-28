# Industrial PPE Detection (YOLO + real-time)

import os
from ultralytics.models import YOLO
import torch

if __name__=='__main__':
    model=YOLO(model='yolo26n.pt')
    print('CUDA:',torch.cuda.is_available())

    #fine tunning:  
    result=model.train(
    data="D:\Courses\dataset\PPE_Detection_datafolder\data.yaml",
    epochs=5,
    batch=4,
    workers=0,
    imgsz=720,
    device='cuda',
    project='project/trained_model'     
)
    print('Done Training')