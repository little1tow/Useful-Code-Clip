# some necessary package
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import numpy as np 


# build own network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # 灰度图像的channels=1即in_channels=1 输出为10个类别即out_features=10
        # parameter(形参)=argument(实参) 卷积核即卷积滤波器 out_channels=6即6个卷积核 输出6个feature-maps(特征映射)
        # 权重shape 6*1*5*5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)  # 二维批归一化 输入size=6
        # 权重shape 12*1*5*5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        # 全连接层：fc or dense or linear out_features即特征(一阶张量)
        # 权重shape 120*192
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.bn2 = nn.BatchNorm1d(120)  # 一维批归一化 输入size=120
        # 权重shape 60*120
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        # 权重shape 10*60
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        # (1) input layer
        t = t
        # (2) hidden conv layer
        t = F.relu(self.conv1(t))  # (28-5+0)/1+1=24 输入为b(batch_size)*1*28*28 输出为b*6*24*24 relu后shape不变
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # (24-2+0)/2+1=12 输出为b*6*12*12
        t = self.bn1(t)
        
        # (3) hidden conv layer
        t = F.relu(self.conv2(t))  # (12-5+0)/1+1=8 输出为b*12*8*8 relu后shape不变
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # (8-2+0)/2+1=4 输出为b*12*4*4
        
        # (4) hidden linear layer
        t = F.relu(self.fc1(t.reshape(-1, 12*4*4)))  # t.reshape后为b*192 全连接层后输出为b*120 relu后shape不变
        t = self.bn2(t)
        # (5) hidden linear layer
        t = F.relu(self.fc2(t))  # 全连接层后输出为b*60 relu后shape不变
        
        # (6) output layer
        t = self.out(t)  # 全连接层后输出为b*10 relu后shape不变
        return t


# data processing
class ProcessImgAndGt(object):
    def __init__(self, transforms):
            self.transforms = transforms
    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
    def __call__(self, img, label):
        img = img.resize((self.width, self.height), Image.BILINEAR)
        label = label.resize((self.width, self.height), Image.NEAREST)
        return img, label

class Normalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std
    def __call__(self, img, label):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img, label

class ToTensor(object):
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()
    def __call__(self, img, label):
        img, label = self.to_tensor(img), self.to_tensor(label).long()
        return img, label


transforms = ProcessImgAndGt([
    Resize(512, 512),
    Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1]),
    ToTensor()
])


# dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transforms):
        super(TrainDataset, self).__init()
        self.dataset_path = dataset_path
        self.transforms = transforms
        # 根据具体的业务逻辑读取全部数据路径作为加载数据的索引
        for dir in os.listdir(dataset_path):
            image_dir = os.path.join(dataset_path, dir)
            gt_path = image_dir + '/GT/'
            img_path = image_dir + '/Frame/'
            img_list = []
            for name in os.listdir(img_path):
                if name.endswith('.png'):
                    img_list.append(name)
            self.file_list.extend([(img_path + name, gt_path + name) for name in img_list])

    def __getitem__(self, idx):   
        img_path, label_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        img, label = self.transforms(img, label)
        return img, label

    def __len__(self):
        return len(self.file_list)

# optimizer
base_params = [params for name, params in model.named_parameters() if ("xxx" in name)]
finetune_params = [params for name, params in model.named_parameters() if ("yyy" in name)]
optimizer = optim.Adam([
    {"params": base_params},
    {"params": finetune_params, "lr": 1e-3}
], lr=1e-4, weight_decay=1e-4);


# run the model
model = Network().cuda()
# 构建数据预处理
transforms = ProcessImgAndGt([
    Resize(512, 512),
    Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1]),
    ToTensor()
])
# 构建Dataset
train_dataset = MyDataset(train_dataset_path, transforms)
# DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=12,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   pin_memory=False)
# TestDataset
test_dataset = MyDataset(test_dataset_path, transforms)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=4,
                                                  shuffle=True,
                                                  num_workers=2,
                                                  pin_memory=False)

# optimizer需要传入全部需要更新的参数名称，这里是对不用的参数执行不同的更新策略 
base_params = [params for name, params in model.named_parameters() if ("xxx" in name)]
finetune_params = [params for name, params in model.named_parameters() if ("yyy" in name)]
optimizer = torch.optim.Adam([
    {"params": base_params, "lr": 1e-3, ...},
    {"params": finetune_params, "lr": 1e-4, ...}
])

for epoch in range(20):
    model.train()
    epoch_loss = 0
    for batch in trian_loader:
        images. gts = batch[0].cuda(), batch[1].cuda()
        preds = model(iamges)
        loss = F.cross_entropy(preds, gts)
        optimizer.zero_grad()    # pytorch会积累梯度，在优化每个batch的权重的梯度之前将之前计算出的每个权重的梯度置0
        loss.backward()          # 在最后一个张量上调用反向传播方法，在计算图中计算权重的梯度 
        optimizer.step()         # 使用预先设置的学习率等参数根据当前梯度对权重进行更新
        epoch_loss += loss * trian_loader.batch_size
        # 计算其他标准
    loss = epoch_loss / len(train_loader.dataset)
    # .......
    # 每隔几个epoch在测试集上跑一下
    if epoch % 5 == 0:
        model.eval()
        test_epoch_loss = 0
        for test_batch in test_loader:
            test_images. test_gts = test_batch[0].cuda(), test_batch[1].cuda()
            test_preds = model(test_iamges)
            loss = F.cross_entropy(test_preds, test_gts)
            test_epoch_loss += loss * test_loader.batch_size
            # 计算其他标准
        test_loss = test_epoch_loss / (len(test_loader.dataset))
    # .......
    # 根据条件对指定epoch的模型进行保存 将模型序列化到磁盘的pickle包
    if 精度最高:
        torch.save(model.stat_dict(), f'{model_path}_{time_index}.pth')


# model test
test_dataset = MyDataset(test_dataset_path, transforms)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       num_workers=2)
model = Network().cuda()
# 对磁盘上的pickle文件进行解包 将gpu训练的模型加载到cpu上
model.load_stat_dict(torch.load(model_path, map_location=torch.device('cpu')));
mocel.eval()

with torch.no_grad():
    for batch in test_loader:
        test_images. test_gts = test_batch[0].cuda(), test_batch[1].cuda()
        test_preds = model(test_iamges)
        # 保存模型输出的图片