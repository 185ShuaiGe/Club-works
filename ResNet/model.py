import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    expansion = 1   #表示输入和输出通道数相同

    def __init__(self, in_channel, out_channel, stride = 1, downsample = None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace = False)

        self.conv2 = nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:                        # 下采样，对应虚线残差结构，当输入和输出通道数不同时，需要下采样
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # print("Before add - out shape:", out.shape, "identity shape:", identity.shape)
        # assert out.shape == identity.shape, "维度不匹配！"
        #测试过了，维度匹配

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4               #第三层的卷积核个数是第一层的4倍，对应虚线残差结构，当输入和输出通道数不同时，需要下采样
    def __init__(self, in_channels, plains, stride = 1, downsample = None):
        super().__init__()

        target_channels = plains * self.expansion

        self.conv1 = nn.Conv2d(in_channels, plains, kernel_size = 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(plains)

        self.conv2 = nn.Conv2d(in_channels = plains, out_channels = plains, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(plains)

        self.conv3 = nn.Conv2d(in_channels = plains, out_channels = target_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(target_channels)

        self.relu = nn.ReLU(inplace = False)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # print("Before add - out shape:", out.shape, "identity shape:", identity.shape)
        # assert out.shape == identity.shape, "维度不匹配！"

        #测试过了，维度匹配
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, 
                 block,                     #BasicBlock or Bottleneck
                 blocks_num,                #列表参数，所使用残差结构的数目，如对ResNet-34来说即是[3,4,6,3]
                 num_classes = 5,            #分类数
                 include_top = True):        #是否包含全连接层
        super(ResNet, self).__init__()
        self.include_top = include_top

        self.in_channel = 64                # 通过max pooling之后所得到的特征矩阵的深度

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        
        self.relu = nn.ReLU(inplace = False)            # inplace = False 表示不改变输入张量的内存位置，而是创建一个新的张量来存储输出

        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])              # 对应conv2_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)   # 对应conv3_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)   # 对应conv4_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)   # 对应conv5_x

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # 全局平均池化层，output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():                    #self.modules:返回模型中所有子模块的迭代器，包含当前模型和其所有嵌套的子模块
            if isinstance(m, nn.Conv2d):            #判断当前模块是否为二维卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')          #使用Kaiming正态分布初始化卷积层的权重
        #     m.weight: 卷积层权重参数张量，直接对其进行原地初始化（带下划线函数表示会修改原变量）。
        #     mode='fan_out': 计算初始化时考虑的是卷积核输出通道的数量，适合卷积层。
        #     nonlinearity='relu': 告诉初始化函数，后面会接ReLU激活函数，He初始化是在考虑ReLU非线性后设计的初始化策略。

    def _make_layer(self, block, plains, block_num, stride = 1):
        # block即BasicBlock/Bottleneck
        # plains即瓶颈层被压缩之后的通道数
        # block_num即该层一共包含了多少层残差结构

        downsample = None
        target_channels = plains * block.expansion

        # 左：输出的高和宽相较于输入会缩小；右：输入channel数与输出channel数不相等，两者都会使x和identity无法相加
        if stride != 1 or self.in_channel != target_channels:
            # 对于ResNet-50/101/152：
            # conv2_x第一层也是虚线残差结构，但只调整特征矩阵深度，高宽不需调整
            # conv3/4/5_x第一层需要调整特征矩阵深度，且把高和宽缩减为原来的一半
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, target_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(target_channels))
        
        layers = []
        layers.append(block(self.in_channel,                # 输入特征矩阵深度64
                            plains,                        # 残差结构所对应主分支上的第一个卷积层的卷积核个数
                            stride = stride, 
                            downsample = downsample))
        self.in_channel = target_channels                    # 更新输入特征矩阵深度

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, plains))

        # 通过非关键字参数的形式传入nn.Sequential
        return nn.Sequential(*layers)   # *加list或tuple，可以将其转换成非关键字参数，将刚刚所定义的一切层结构组合在一起并返回
    
    def forward(self, x):
        x = self.conv1(x)                   # 7 * 7 卷积层
        x = self.bn1(x)
        x = self.relu(x)                    
        x = self.maxpool(x)                 # 3 * 3 max pooling层

        x = self.layer1(x)                  # conv2_x
        x = self.layer2(x)                  # conv3_x
        x = self.layer3(x)                  # conv4_x
        x = self.layer4(x)                  # conv5_x

        if self.include_top:
            x = self.avgpool(x)             # 全局平均池化层，output size = (1, 1)
            x = torch.flatten(x, 1)         # 展平操作，将特征矩阵展平成一维向量
            x = self.fc(x)                  # 全连接层，output size = num_classes

        return x
    

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])
