import torch
import custom_data_processing
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from Models import HandPoseVGG16
import time



JOINT_LIST_NYU = [31, 28, 23, 17, 11, 5, 27, 22, 16, 10, 4, 25, 20, 14, 8, 2, 24, 18, 12, 6, 0]
ROOT = "/Volumes/Evan_Samsung_HP_data/nyu_dataset/data"

dataset = custom_data_processing.HandposeDataset(ROOT, JOINT_LIST_NYU)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
check_it_out = HandPoseVGG16()
trainable_params = [p for p in check_it_out.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
start = time.time()
for epoch in range(2):
    print("check it out")
    print(f"{time.time() - start} seconds")
    start = time.time()
    check_it_out.train()
    for im, tr in dataloader:
        preds = check_it_out(im)
        loss = loss_fn(preds, tr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


print("all finished")