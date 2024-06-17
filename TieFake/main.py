import sys
import os
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms
from transformers import BertForSequenceClassification
from resnest101 import resnest101_2way
from dataloader import Hybrid_Dataset, my_collate
from models import FusionModel
from trainer import ModelTrainer
import warnings  # 导入警告模块
warnings.filterwarnings("ignore")  # 忽略所有警告

# 设置设备
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

folder = "v3-4*" # 当前训练的数据集
generated = False # 是否使用LLM生成的新数据集
if generated:
    folder_generated = folder + "_generated"
else:
    folder_generated = folder
bert_classifier = BertForSequenceClassification.from_pretrained(f'./Data/{folder_generated}/bert_save_dir')
local_weight_path ="./resnest_model/resnest101-22405ba7.pth"
resnest_model = resnest101_2way(pretrained=True, local_weight_path=local_weight_path)

# 加载已保存的模型权重
hybrid_model = FusionModel(resnest_model, bert_classifier, device)
hybrid_model = hybrid_model.to(device)

csv_dir = f"./Data/{folder}/"
img_dir = "./Data/gossipcop_images/"
l_datatypes = ['train', 'val', 'test']
csv_fnames = {
    'train': 'gossipcop_train.tsv',
    'val': 'gossipcop_val.tsv',
    'test': 'gossipcop_test.tsv'
}
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

if __name__ == '__main__':
    hybrid_datasets = {x: Hybrid_Dataset(os.path.join(csv_dir, csv_fnames[x]), img_dir, transform=data_transforms)
                    for x in l_datatypes}
    dataset_sizes = {x: len(hybrid_datasets[x]) for x in l_datatypes}

    dataloaders = {x: torch.utils.data.DataLoader(hybrid_datasets[x], batch_size=16, shuffle=True, num_workers=2,
                                                collate_fn=my_collate) for x in l_datatypes}
    criterion = nn.BCEWithLogitsLoss()

    optimizer_ft = optim.Adam(hybrid_model.parameters(), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    trainer = ModelTrainer(l_datatypes, hybrid_datasets, dataloaders, hybrid_model)

    trainer.train_model(criterion, optimizer_ft, exp_lr_scheduler, device, num_epochs=10, report_len=1000)
    trainer.save_model(f'./Data/{folder_generated}/result/hybrid_model.pt')
    trainer.generate_eval_report(f'./Data/{folder_generated}/result/hybrid_report.json')