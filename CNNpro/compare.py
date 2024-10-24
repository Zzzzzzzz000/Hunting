import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 数据预处理和加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# 定义原始CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


# 定义CBAM模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# 定义包含CBAM的CNN模型
class CNNWithCBAM(nn.Module):
    def __init__(self):
        super(CNNWithCBAM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.cbam1 = CBAM(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.cbam2 = CBAM(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.cbam3 = CBAM(64)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.cbam1(x)
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = self.cbam2(x)
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = self.cbam3(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


# 训练和评估函数
def train_and_evaluate(model, trainloader, testloader, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(trainloader))

        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        test_losses.append(test_loss / len(testloader))
        accuracy = correct / len(testset)
        accuracies.append(accuracy)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}, Accuracy: {accuracy}')

    return train_losses, test_losses, accuracies


# 训练原始CNN
model = CNN()
train_losses_cnn, test_losses_cnn, accuracies_cnn = train_and_evaluate(model, trainloader, testloader)

# 训练包含CBAM的CNN
model_cbam = CNNWithCBAM()
train_losses_cbam, test_losses_cbam, accuracies_cbam = train_and_evaluate(model_cbam, trainloader, testloader)

# 可视化结果
epochs = range(1, 6)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses_cnn, label='Training Loss - CNN')
plt.plot(epochs, test_losses_cnn, label='Test Loss - CNN')
plt.plot(epochs, train_losses_cbam, label='Training Loss - CNN + CBAM')
plt.plot(epochs, test_losses_cbam, label='Test Loss - CNN + CBAM')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracies_cnn, label='Accuracy - CNN')
plt.plot(epochs, accuracies_cbam, label='Accuracy - CNN + CBAM')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
