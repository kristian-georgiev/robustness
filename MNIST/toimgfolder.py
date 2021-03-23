import os
import torch
from torchvision import transforms


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


x_tr, y_tr = torch.load('processed/training.pt')
x_ts, y_ts = torch.load('processed/training.pt')

idx_tr = ((y_tr == 3) + (y_tr == 5))
idx_ts = ((y_ts == 3) + (y_ts == 5))

x_tr = x_tr[idx_tr]
y_tr = y_tr[idx_tr]

x_ts = x_ts[idx_ts]
y_ts = y_ts[idx_ts]

topil = transforms.ToPILImage(mode='L')

for i in range(len(y_tr)):
    img = topil(x_tr[i])
    label = str(y_tr[i].item())
    path = f"./img_folder/train/{label}/img{str(i)}.png"
    ensure_dir(path)
    img.save(path)

for i in range(len(y_ts)):
    img = topil(x_ts[i])
    label = str(y_ts[i].item())
    path = f"./img_folder/val/{label}/img{str(i)}.png"
    ensure_dir(path)
    img.save(path)
