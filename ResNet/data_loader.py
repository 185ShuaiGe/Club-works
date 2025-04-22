# data_loader.py
import os
import torch
import json
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_train_loader(batch_size=16):
    # 数据增强
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
    

    #从当前目录读取数据并加载数据集
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join( "D:\VScode\Code\VScode程序\俱乐部小作业\ResNet", "data_set", "flowers")  # flower data set path

    print("----------------------------", image_path, "----------------------------")

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])

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
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=0)

    return train_loader, validate_loader

if __name__ == '__main__':
    train_loader, validate_loader = get_train_loader()