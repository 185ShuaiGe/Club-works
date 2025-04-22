# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from model import resnet34, resnet50, resnet101
from data_loader import get_train_loader

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_loader, validate_loader = get_train_loader()

# %%
# 加载预处理好的模型参数
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

# %%
# 可视化训练过程中的损失和准确率
def visualize_performance(train_loss, val_loss, val_acc, name):
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
    plt.savefig('./visual_result/' + name + '.png')
    plt.show()


# %%
# 构造模型
def model(net, name, loss_function, train_loader, validate_loader, save_path, lr = 0.0001, epoch = 5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)             #实例化，这里没有传入参数num_classes，即实例化后的最后一个全连接层有1000个节点

    best_acc = 0.0   
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    # construct an optimizer
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = 1e-4)
    # 添加学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(epoch):
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

                validate_loader.desc = "valid epoch[{}/{}]".format(epoch + 1, 3)
        epoch_val_loss = val_loss / len(validate_loader)
        val_loss_history.append(epoch_val_loss)
        val_accurate = acc / len(validate_loader)
        val_acc_history.append(val_accurate)
        print('[epoch %d] train_loss: %.3f  val_loss: %.3f val_accuracy: %.3f' %
            (epoch + 1, epoch_train_loss, epoch_val_loss, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

    # 训练结束后调用
    visualize_performance(train_loss_history, val_loss_history, val_acc_history, name)

# %%
# train the model
# define loss function
loss_function = nn.CrossEntropyLoss()

save_path = './weight_outcome'

# %%
net1 = resnet34(num_classes = 5, include_top = True)
model(net1, 'Resnet34', loss_function , train_loader, validate_loader, save_path = save_path + '/ResNet34.pth', lr = 0.001, epoch = 20)

# %%
net2 = resnet50(num_classes = 5, include_top = True)
model(net2, 'Resnet50', loss_function, train_loader, validate_loader, save_path = save_path + '/ResNet50.pth', lr = 0.0001, epoch = 50)

# %%
net3 = resnet101(num_classes = 5, include_top = True)
model(net3, 'Resnet101', loss_function, train_loader, validate_loader, save_path = save_path + '/ResNet101.pth', lr = 0.0001, epoch = 50)


