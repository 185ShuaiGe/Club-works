import torch
from model import BasicBlock, Bottleneck, ResNet
from data_loader import get_train_loader

train_loader, validate_loader = get_train_loader()

# ---------------------------
# 1. 提取少量数据 (1个批量)
# ---------------------------
def extract_sample(train_loader):
    data_iter = iter(train_loader)              # 创建一个迭代器, 使用next()获取一个批量的数据
    images, _ = next(data_iter)                 # 提取第一个批量（16张图），忽略标签
    return images

# ---------------------------
# 2. 调试维度专用模型（继承你的ResNet，添加维度打印）
# ---------------------------
class DebugResNet(ResNet):
    def forward(self, x):
        print("\nInput to conv1:", x.shape)
        x = self.conv1(x)
        print("After conv1:", x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print("After maxpool:", x.shape)
        
        # 遍历每个残差层
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            print(f"\nEntering layer {i+1}")
            x = layer(x)
            print(f"After layer {i+1}:", x.shape)
        
        if self.include_top:
            x = self.avgpool(x)
            print("After avgpool:", x.shape)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            print("After fc:", x.shape)
        return x

# ---------------------------
# 3. 初始化调试模型（以ResNet-50为例）
# ---------------------------
def create_debug_model():
    # 注意：此处使用修正后的Bottleneck和_make_layer逻辑
    model = DebugResNet(
        block=Bottleneck,
        blocks_num=[3, 4, 6, 3],
        num_classes=5,          # 假设你的分类数是5
        include_top=True
    )
    return model

# ---------------------------
# 4. 运行验证
# ---------------------------
def validate_dimensions(train_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 提取数据
    sample_images = extract_sample(train_loader)
    sample_images = sample_images.to(device)
    print("\n输入数据维度:", sample_images.shape)  # 应为 [16, 3, 224, 224]
    
    # 初始化模型
    model = create_debug_model().to(device)
    
    # 前向传播并打印维度
    with torch.no_grad():
        _ = model(sample_images)

# ---------------------------
# 执行脚本（在你的训练代码中调用）
# ---------------------------
if __name__ == "__main__":
    # 假设 train_loader 已定义（根据你的训练代码）
    validate_dimensions(train_loader)