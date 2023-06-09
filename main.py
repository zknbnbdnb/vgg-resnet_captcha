import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from dataset import train_loader,val_loader
import matplotlib.pyplot as plt
import numpy as np

#设置工作路径
os.chdir("D:/pytorch/Verification_code")

#超参数的设置

# 学习率
lr = 1e-3
# 最大迭代次数
MAX_EPOCH = 200
# 是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义训练模型
def train(net):
    # 读入数据，这边的注意事项就是需要打乱数据
    # 选用Adam优化器
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # 选用交叉熵作为损失函数，默认情况下reduction="elementwise_mean",这里选择none输出则是向量
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss_records = []
    # 训练模式
    net.train()
    for epoch_idx in range(MAX_EPOCH):
        # 记录每个epoch的正确个数
        correct_cnt = 0
        # 记录每个epoch的损失
        ovr_loss = 0.
        for batch_idx, (img, label) in enumerate(train_loader):
            # 将数据放入GPU
            img, label = img.to(device), label.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            hint = net(img)
            # 计算损失
            loss = criterion(hint, label)
            # 反向传播
            loss.mean().backward()
            # 更新参数
            optimizer.step()
            # 计算正确个数
            pred = torch.argmax(hint, dim=-1)
            correct_cnt += ((pred == label).int().sum()).item()
            # 计算损失
            ovr_loss += loss.sum().item()
        # 计算准确率
        acc = correct_cnt / len(train_loader.dataset)
        # 计算平均损失
        mean_loss = ovr_loss / len(train_loader.dataset)
        # 记录损失
        loss_records.append(mean_loss)
        print('训练 Epoch: {}/{} 准确率: {:.2f}% Loss值: {:.6f}'.
                format(epoch_idx + 1, MAX_EPOCH, 100 * acc, mean_loss))

    return loss_records

def validate(net):
    # 测试模式
    net.eval()
    with torch.no_grad():
        # 记录正确个数
        correct_cnt = 0
        for img, label in val_loader:
            # 将数据放入GPU
            img, label = img.to(device), label.to(device)
            # 前向传播
            hint = net(img)
            # 计算正确个数
            pred = torch.argmax(hint, dim=-1)
            correct_cnt += ((pred == label).int().sum()).item()
    # 计算准确率
    acc = correct_cnt / len(val_loader.dataset)
    print("测试集上的准确率为 {:.2f}".format(acc))


# 绘制最后的损失函数图像
def draw_loss_figure(loss_records, title="loss", imgname="loss.png"):
    x = np.arange(0, len(loss_records))
    plt.plot(x, loss_records, color='red', linewidth=1)
    plt.xticks(np.arange(0, len(loss_records)))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(imgname)
    print("loss曲线已保存到{}".format(imgname))
    plt.close()


# 绘制最后的损失函数图像
def draw_loss_figures(loss_records, title="loss"):
    # colors = ['red', 'yellow', 'blue', 'cyan']

    plt.figure()
    for (loss_record, nm) in loss_records:
        x = np.arange(0, len(loss_record))
        plt.plot(x, loss_record, label=nm)
        plt.xticks(np.arange(0, len(loss_record)))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig("All losses.png")
    print("All loss曲线已保存到All losses.png")


if __name__ == "__main__":
    loss_records_list = []
    # 这里作为一个选择，因为以后可能有更多的backbone，那么我们只需要加入到此列表即可
    for backbone_type in ['resnet', 'vgg']:
        print("使用{}网络".format(backbone_type))
        # 选择网络
        net = get_model(backbone_type).to(device)
        # 训练网络
        loss_records = train(net)
        # 测试网络
        validate(net)
        # 保存网络
        torch.save(net.state_dict(), '{}.tar'.format(backbone_type))
        loss_records_list.append((loss_records, backbone_type))
        draw_loss_figure(loss_records,
                        title="{} loss".format(backbone_type),
                        imgname="{}_loss.png".format(backbone_type))
    draw_loss_figures(loss_records_list, "All losses")