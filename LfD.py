import os
from PIL import Image
import time
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import numpy as np
import datasets_fF_testset
from resnet_cifar_mini import *
import torchvision.transforms as transforms
import torch.utils.data as Data
import random
from vgg_cifar import vgg_s,vgg19,vgg16_bn

class augDataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        index = self.dataset[index][2]


        # 对所有数据进行数据增强
        if self.transform:
            image = self.transform(image)
        return index, image, label

    def __len__(self):
        return self.dataLen

tf_train = transforms.Compose([
     transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

tf_noaug = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


batch_size = 128
alpha = 0.0003
threshold = 1
device = 'cuda'

# 这段是原本从npy文件读取各种数据集的，记得之后解掉注释
dataset = torch.load(r'C:\Users\Lee\Desktop\backdoor-toolbox-main\cifar10_poisoned_gridTrigger_1.pth')
train_dataset = augDataset_npy(full_dataset=dataset, transform=tf_train)
train_dataset_noaug = augDataset_npy(full_dataset=dataset, transform=tf_noaug)
print("lenth of traindata:"+str(len(train_dataset)))
train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
)


_,valid_loader,valid_loader_BD = datasets_fF_testset.get_loader('CIFAR10',inject_portion = 0, target_label = 0, \
trig_w=2,trig_h=2,triggerType='gridTrigger', target_type="all2one")

model_b = resnet6(num_classes=10, norm_layer=nn.BatchNorm2d)
model_d = resnet18(num_classes=10)

model_b = model_b.to(device)
model_d = model_d.to(device)

checkpoint = torch.load(
    'lfd_teacher/lfdmodel_state_dict_cifar10_grid.pth',
    #'./lfd_teacher/lfdmodel_state_dict_cifar100_.pth',
    map_location=device)
model_b.load_state_dict(checkpoint['state_dict'])

model_b.eval()

optimizer_d = torch.optim.Adam(
    model_d.parameters(),
    lr=1e-3,
    weight_decay=0.0,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=81)
criterion_batch = nn.CrossEntropyLoss(reduction='none')
criterion = nn.CrossEntropyLoss()


def evaluate(net, data_loader, state):
    global best_acc
    net.eval()
    test_loss_bd = 0
    correct_bd = 0
    total_bd = 0
    acc_bd = 0
    acc_ori = 0
    list_loss = []
    with torch.no_grad():
        for batch_idx, (indexs, inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
            criterion2 = nn.CrossEntropyLoss()
            loss = criterion2(outputs, targets)
            criterion_batch = nn.CrossEntropyLoss(reduction='none')
            with torch.no_grad():
                loss_sample = criterion_batch(outputs, targets)
            for i in loss_sample:
                list_loss.append(i)

            test_loss_bd += loss.item()
            _, predicted = outputs.max(1)

            total_bd += targets.size(0)
            correct_bd += predicted.eq(targets).sum().item()
            acc_bd = acc_bd + 100. * correct_bd / total_bd

    loss = test_loss_bd / (batch_idx + 1)
    acc = 100. * correct_bd / total_bd

    if state == "train":
        net.train()
    else:
        state = "placeholder"
    return acc, loss, list_loss

loss_weight = []
with torch.no_grad():
    for i in train_dataset_noaug:
        _,logit_noaug = model_b(i[1].unsqueeze(dim=0).to(device))
        loss_weight.append(criterion_batch(logit_noaug, torch.tensor(i[2]).unsqueeze(dim=0).to(device)))


# 将权重减去阈值，再将小于0的权重乘以系数。基础设置
loss_weight = torch.tensor(loss_weight)


len_loss_weight = len(loss_weight)
loss_weight_max = loss_weight.max()
loss_weight = loss_weight / loss_weight_max
loss_weight_sort = loss_weight.argsort()
loss_threshold = loss_weight[loss_weight_sort[int(len_loss_weight*0.15)]]
loss_weight_weight = loss_weight-loss_threshold
loss_weight_weight /= loss_weight_weight.max()

for weight in loss_weight_weight:
    if weight<=0:
        weight *= 0.001

start = time.time()

for step in tqdm(range(int(50000 / batch_size) * (81))):
    # train main model
    try:
        index, data, attr = next(train_iter)
    except:
        train_iter = iter(train_loader)
        index, data, attr = next(train_iter)

    data = data.to(device)
    attr = attr.to(device)
    label = attr.to(device)
    _, logit_d = model_d(data)

    # 这段为使用loss过滤样本

    loss_d_all = criterion_batch(logit_d, label)

    # 正常训练时去掉这句
    loss_d_weight = loss_weight_weight[index]
    loss_d_all = loss_d_all * loss_d_weight.to(device)

    optimizer_d.zero_grad()
    loss = loss_d_all.mean()
    loss.backward()
    optimizer_d.step()

    if (step + 1) % int(50000 / batch_size + 1) == 0:
        #valid_attrwise_accs_d_BD, loss_BD, list_loss_BD_d = evaluate(model_d, valid_loader_BD, state="train")
        #valid_attrwise_accs_d_ORI, loss_ORI, list_loss_ORI_d = evaluate(model_d, valid_loader, state="train")
        #valid_attrwise_accs_d_train, loss_train, list_loss_train = evaluate(model_d, train_loader, state="train")
        #valid_attrwise_accs_b_BD, loss_b_BD, list_loss_BD_b = evaluate(model_b, valid_loader_BD, state="test")
        #valid_attrwise_accs_b_ORI, loss_b_ORI, list_loss_ORI_b = evaluate(model_b, valid_loader, state="test")
        #print("ACC train debias: " + str(valid_attrwise_accs_d_train))
        #print("LOSS train debias: " + str(loss_train))
        #print("ACC BD debias: " + str(valid_attrwise_accs_d_BD))
        #print("ACC ORI debias: " + str(valid_attrwise_accs_d_ORI))
        #print("ACC BD bias: " + str(valid_attrwise_accs_b_BD))
        #print("ACC ORI bias: " + str(valid_attrwise_accs_b_ORI))
        #print("LOSS BD debias: " + str(loss_BD))
        #print("LOSS ORI debias: " + str(loss_ORI))

        scheduler.step()
        #print("lr: " + str(scheduler.get_last_lr()[0]))
    if (step + 1) % int(500000 / batch_size + 1) == 0:
        valid_attrwise_accs_d_BD, loss_BD, list_loss_BD_d = evaluate(model_d, valid_loader_BD, state="train")
        valid_attrwise_accs_d_ORI, loss_ORI, list_loss_ORI_d = evaluate(model_d, valid_loader, state="train")
            # valid_attrwise_accs_d_train, loss_train, list_loss_train = evaluate(model_d, train_loader, state="train")
        #valid_attrwise_accs_b_BD, loss_b_BD, list_loss_BD_b = evaluate(model_b, valid_loader_BD, state="test")
        #valid_attrwise_accs_b_ORI, loss_b_ORI, list_loss_ORI_b = evaluate(model_b, valid_loader, state="test")
            # print("ACC train debias: " + str(valid_attrwise_accs_d_train))
            # print("LOSS train debias: " + str(loss_train))
        print("ACC BD debias: " + str(valid_attrwise_accs_d_BD))
        print("ACC ORI debias: " + str(valid_attrwise_accs_d_ORI))
        #print("ACC BD bias: " + str(valid_attrwise_accs_b_BD))
        #print("ACC ORI bias: " + str(valid_attrwise_accs_b_ORI))
            # print("LOSS BD debias: " + str(loss_BD))
            # print("LOSS ORI debias: " + str(loss_ORI))

valid_attrwise_accs_d_BD, loss_BD, list_loss_BD_d = evaluate(model_d, valid_loader_BD, state="train")
valid_attrwise_accs_d_ORI, loss_ORI, list_loss_ORI_d = evaluate(model_d, valid_loader, state="train")
#valid_attrwise_accs_d_train, loss_train, list_loss_train = evaluate(model_d, train_loader, state="train")
#valid_attrwise_accs_b_BD, loss_b_BD, list_loss_BD_b = evaluate(model_b, valid_loader_BD, state="test")
#valid_attrwise_accs_b_ORI, loss_b_ORI, list_loss_ORI_b = evaluate(model_b, valid_loader, state="test")
#print("ACC train debias: " + str(valid_attrwise_accs_d_train))
#print("LOSS train debias: " + str(loss_train))
print("ACC BD debias: " + str(valid_attrwise_accs_d_BD))
print("ACC ORI debias: " + str(valid_attrwise_accs_d_ORI))
#print("ACC BD bias: " + str(valid_attrwise_accs_b_BD))
#print("ACC ORI bias: " + str(valid_attrwise_accs_b_ORI))
#print("LOSS BD debias: " + str(loss_BD))
#print("LOSS ORI debias: " + str(loss_ORI))
checkpoint = {
    'state_dict': model_b.state_dict(),
}

torch.save(checkpoint, 'final_lfd/result_grid_cifar10.pth')