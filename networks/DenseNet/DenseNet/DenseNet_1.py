import os
import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time

 
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
 
batch_size = 32
path = './'
train_transform = transforms.Compose([
    transforms.RandomSizedCrop(96),# 随机剪切成227*227
    transforms.RandomHorizontalFlip(),# 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])
val_transform = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])
 
traindir = os.path.join(path, 'train')
valdir = os.path.join(path, 'val')
 
train_set = torchvision.datasets.CIFAR10(
    traindir, train=True, transform=train_transform, download=True)
valid_set = torchvision.datasets.CIFAR10(
    valdir, train=False, transform=val_transform, download=True)
 
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
 
dataloaders = {
    'train': train_loader,
    'valid': valid_loader,
#     'test': dataloader_test
    }
 
dataset_sizes = {
    'train': len(train_set),
    'valid': len(valid_set),
#     'test': len(test_set)
    }
print(dataset_sizes)
 
def train(model, criterion, optimizer, scheduler, device, num_epochs, dataloaders,dataset_sizes):
    model = model.to(device)
    print('training on ', device)
    since = time.time()
 
    best_model_wts = []
    best_acc = 0.0
 
    for epoch in range(num_epochs):
        # 训练模型
        s = time.time()
        model,train_epoch_acc,train_epoch_loss = train_model(
            model, criterion, optimizer, dataloaders['train'], dataset_sizes['train'], device)
        print('Epoch {}/{} - train Loss: {:.4f}  Acc: {:.4f}  Time:{:.1f}s'
            .format(epoch+1, num_epochs, train_epoch_loss, train_epoch_acc,time.time()-s))
        # 验证模型
        s = time.time()
        val_epoch_acc,val_epoch_loss = val_model(
            model, criterion, dataloaders['valid'], dataset_sizes['valid'], device)
        print('Epoch {}/{} - valid Loss: {:.4f}  Acc: {:.4f}  Time:{:.1f}s'
            .format(epoch+1, num_epochs, val_epoch_loss, val_epoch_acc,time.time()-s))
        # 每轮都记录最好的参数.
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = model.state_dict()
        # 优化器
#         if scheduler not in None:
#             scheduler.step()
        # 保存画图参数
        train_losses.append(train_epoch_loss.to('cpu'))
        train_acc.append(train_epoch_acc.to('cpu'))
        val_losses.append(val_epoch_loss.to('cpu'))
        val_acc.append(val_epoch_acc.to('cpu'))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # model.load_state_dict(best_model_wts)
    return model
 
def train_model(model, criterion, optimizer, dataloader, dataset_size,device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs,labels in dataloader:
        optimizer.zero_grad()
        # 输入的属性
        inputs = Variable(inputs.to(device))
        # 标签
        labels = Variable(labels.to(device))
        # 预测
        outputs = model(inputs)
        _,preds = torch.max(outputs.data,1)
        # 计算损失
        loss = criterion(outputs,labels)
        #梯度下降
        loss.backward()
        optimizer.step()
 
        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data)
 
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    
    return model,epoch_acc,epoch_loss
 
def val_model(model, criterion, dataloader, dataset_size, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for (inputs,labels) in dataloader:
        # 输入的属性
        inputs = Variable(inputs.to(device))
        # 标签
        labels = Variable(labels.to(device))
        # 预测
        outputs = model(inputs)
        _,preds = torch.max(outputs.data,1)
        # 计算损失
        loss = criterion(outputs,labels)
        
        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data)
 
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    
    return epoch_acc,epoch_loss
class DenseNet(nn.Module):
 
    def __init__(self):
        super().__init__()
 
        self.b1 = nn.Sequential(
                  nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                  nn.BatchNorm2d(64), nn.ReLU(),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
       # num_channels为当前的通道数
        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            # 上一个稠密块的输出通道数
            num_channels += num_convs * growth_rate
            # 在稠密块之间添加一个转换层，使通道数量减半
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        # *接受元组形式的参数
        self.conv = nn.Sequential(*blks)
        
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fn = nn.Flatten()
        self.fc = nn.Linear(num_channels, 10)
        
    def forward(self, x):
        out = self.b1(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = self.fn(out)
        out = self.fc(out)
        return out
# 过渡层 --通过1x1的卷积减少通道数，通过平均汇聚层减半高和宽
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), 
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
 
# 一个稠密块由多个卷积块组成，每个卷积块使用相同数量的输出通道。 
# 然而，在前向传播中，我们将每个卷积块的输入和输出在通道维上连结。
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)
 
    def forward(self, x):
        for blk in self.net:
            out = blk(x)
            # 连接通道维度上每个块的输入和输出
            x = torch.cat((x, out), dim=1)
        return x
 
# 稠密块
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), 
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
 
X = torch.randn(1, 3, 96, 96)
net = DenseNet()
net = nn.Sequential(net.b1, net.conv, net.bn, net.relu,
                    net.avgpool,net.fn, net.fc)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
 
val_losses,val_acc = [],[]
train_losses,train_acc = [],[]
model = DenseNet()
 
lr,num_epochs = 0.1, 10
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
 
model = train(model, criterion, optimizer, None ,
              try_gpu(), num_epochs, dataloaders, dataset_sizes)
 
lr,num_epochs = 0.01, 5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
 
model = train(model, criterion, optimizer, None ,
              try_gpu(), num_epochs, dataloaders, dataset_sizes)
 
plt.plot(range(1, len(train_losses)+1),train_losses, 'b', label='training loss')
plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='val loss')
plt.legend()
 
plt.plot(range(1,len(train_acc)+1),train_acc,'b--',label = 'train accuracy')
plt.plot(range(1,len(val_acc)+1),val_acc,'r--',label = 'val accuracy')
plt.legend()