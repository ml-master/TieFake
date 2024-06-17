import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from resnest101 import resnest101_2way
from dataloader import FakedditImageDataset, my_collate
import os, time, copy
from tqdm import tqdm
from collections import deque
from statistics import mean
from torch.autograd import Variable
import ssl
import numpy as np
import warnings  # 导入警告模块
warnings.filterwarnings("ignore")  # 忽略所有警告


def train_model(model, folder, criterion, optimizer, scheduler, num_epochs=1, report_len=20):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in l_datatypes:
            print(f'{phase} phase')
            if phase == 'train':
                model.train() 
            else:
                model.eval() 

            running_loss = 0.0
            running_corrects = 0
            loss_q = deque(maxlen=report_len)
            acc_q = deque(maxlen=report_len)
            counter = 0
            for datas in tqdm(dataloaders[phase]):
                inputs=[]
                labels=[]
                counter += 1
                for i in range(len(datas)):
                    if datas[i] is None:
                        continue
                    inputs.append(datas[i][0])
                for i in range(len(datas)):
                    if datas[i] is None:
                        continue
                    labels.append(datas[i][1])

                labels=torch.tensor(labels)
                labels = labels.to(device)
                inputs=torch.stack(inputs)
                inputs=torch.tensor(inputs)
                inputs = inputs.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    t_pred = pred > 0.5
                    acc = (t_pred.squeeze() == labels).float().sum() / len(labels)
                    acc_q.append(acc.item())
                    loss = criterion(outputs, labels.unsqueeze(-1).float())
                    loss_q.append(loss.item())
                    if counter % report_len == 0:
                        print(f"Iter {counter}, loss: {mean(loss_q)}, accuracy:{mean(acc_q)}")
                    if phase == 'train':
                        loss.requires_grad_(True)
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(t_pred.squeeze() == labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'validate' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f"./Data/{folder}/resnet_best_model_epochs20_full_train")


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return inputs, labels

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # 加载数据
    folder = "v3-4*" # 当前训练的数据集
    
    csv_dir = f"./Data/{folder}/"
    img_dir = "./Data/gossipcop_images/"
    l_datatypes = ['train', 'val']
    csv_fnames = {'train': 'gossipcop_train.tsv', 'val': 'gossipcop_test.tsv'}
    image_datasets = {x: FakedditImageDataset(os.path.join(csv_dir, csv_fnames[x]), img_dir, transform=data_transforms) for x in
                    l_datatypes}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=0,collate_fn=lambda x:x) for x in l_datatypes}
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn) for x in l_datatypes}
    dataset_sizes = {x: len(image_datasets[x]) for x in l_datatypes}
    
    print("Note: corrupted images will be skipped in training")
    local_weight_path ="./resnest_model/resnest101-22405ba7.pth"
    model_ft = resnest101_2way(pretrained=True, local_weight_path=local_weight_path)
    set_parameter_requires_grad(model_ft, True) 

    model_ft = model_ft.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, folder, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    torch.save(model_ft.state_dict(), f'./Data/{folder}/resnest101_epochs10_full_train.pt')
    torch.save(model_ft, f"./Data/{folder}/resnest101_model_save_epochs10_full_train")
