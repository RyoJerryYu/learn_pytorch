import torch
import torch.nn as nn
import torch.nn.functional as F


batch_size = 16
# 批次的大小
lr = 1e-4
# 优化器的学习率
max_epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2 * 2 max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())


def initialize_weights(model: nn.Module):
    for m in model.modules():
        # 判断是否属于Conv2d
        if isinstance(m, nn.Conv2d):
            torch.nn.init.ones_(m.weight.data)
            # 判断是否有偏置
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.3)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0.1)
            if m.bias is not None:
                torch.nn.init.ones_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zeros_()


print(net.conv1.weight.data)
print("Initialize weights")
net.apply(initialize_weights)
print(net.conv1.weight.data)


input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))
