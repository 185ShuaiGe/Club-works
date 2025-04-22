import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from model import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 新增颜色扰动
                                 # 在对图像进行标准化处理时，标准化参数来自于官网所提供的tansfer learning教程
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

    # Resize()函数，输入可能是sequence（元组类型，输入图像高和宽），也可能是int（将最小边缩放到指定的尺寸）
    "val": transforms.Compose([transforms.Resize(256),  # 保持原图片长宽比不变，将最短边缩放到256
                               transforms.CenterCrop(224),  # 中心裁剪一个224×224的图片
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = os.path.join( "D:\VScode\Code\VScode程序\俱乐部小作业\ResNet", "data_set", "flowers")  # flower data set path

print("----------------------------", image_path, "----------------------------")

train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)  # Linux系统把线程个数num_workers设置成＞0，可以加速图像预处理过程

validate_dataset = datasets.ImageFolder(root = os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

net = resnet34()  # 实例化ResNet-34，这里没有传入参数num_classes，即实例化后的最后一个全连接层有1000个节点
net.to(device)

# # load pretrain weights
# # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
# model_weight_path = "./resnet34-pre.pth"    # 保存权重的路径
# missing_keys, unexpected_keys = net.load_state_dict(
#     torch.load(model_weight_path, map_location='cpu'))  # torch.load载入模型权重到内存中（还没有载入到模型中）
# # for param in net.parameters():
# #     param.requires_grad = False
# # change fc layer structure
# in_channel = net.fc.in_features    # 输入特征矩阵的深度
# net.fc = nn.Linear(in_channel, 5)  # 五分类（花分类数据集）

# define loss function
loss_function = nn.CrossEntropyLoss()

# construct an optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay = 1e-4)
# 添加学习率调度
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_acc = 0.0
save_path = './resNet34.pth'

train_loss_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(10):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss:{:^3.0f}%[{}—>{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
    epoch_train_loss = running_loss / len(train_loader)
    train_loss_history.append(epoch_train_loss)
    print()
    scheduler.step()  # 更新学习率(学习率调度)

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    val_loss = 0.0
    with torch.no_grad():
        for val_data in validate_loader:
            test_images, test_labels = val_data
            outputs = net(test_images.to(device))
            loss = loss_function(outputs, test_labels.to(device))

            val_loss += loss.item()

            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

            validate_loader.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       3)
    epoch_val_loss = val_loss / len(validate_loader)
    val_loss_history.append(epoch_val_loss)
    val_accurate = acc / val_num
    val_acc_history.append(val_accurate)
    print('[epoch %d] train_loss: %.3f  val_loss: %.3f val_accuracy: %.3f' %
          (epoch + 1, epoch_train_loss, epoch_val_loss, val_accurate))

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

print('Finished Training')

def visualize_performance(train_loss, val_loss, val_acc):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./training_metrics.png')
    plt.show()

# 训练结束后调用
visualize_performance(train_loss_history, val_loss_history, val_acc_history)