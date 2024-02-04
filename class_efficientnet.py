import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from PIL import Image
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import torchvision.transforms.v2 as T
from efficientnet_pytorch import EfficientNet
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 定義 customdataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = []
        self.labels = []

        for label, character in enumerate(os.listdir(data_dir)):
            character_path = os.path.join(data_dir, character)
            if os.path.isdir(character_path):
                for image_file in os.listdir(character_path):
                    if image_file.endswith(".jpg"):
                        self.image_list.append(os.path.join(character_path, image_file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 定義預訓練模型:EfficientNet model和所使用的預訓練權重
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=50)


model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class AddSpeckleNoise(object):
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_level
        noisy_tensor = tensor * (1 + noise)
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor

class AddPoissonNoise(object):
    def __init__(self, lam=1.0):
        self.lam = lam

    def __call__(self, tensor):
        noise = torch.poisson(self.lam * torch.ones(tensor.shape))
        noisy_tensor = tensor + noise / 255.0
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor

class AddSaltPepperNoise(object):
    def __init__(self, salt_prob=0.05, pepper_prob=0.05):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, tensor):
        noise = torch.rand(tensor.size())
        tensor[(noise < self.salt_prob)] = 1
        tensor[(noise > 1 - self.pepper_prob)] = 0
        return tensor

transform = T.Compose([
    T.ToTensor(),  

    T.RandomApply([T.RandomHorizontalFlip()], p=0.1),
    T.RandomApply([T.RandomVerticalFlip()], p=0.1),
    T.RandomApply([T.RandomRotation(10)], p=0.1),

    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
    T.RandomGrayscale(p=0.1),
    T.RandomInvert(p=0.1),
    T.RandomPosterize(bits=2, p=0.1),
    T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.05),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),

    T.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1),
    T.RandomApply([AddPoissonNoise(lam=0.1)], p=0.1),
    T.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.1),
    T.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.1),

    T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
    T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
    T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.1),

    T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),

    T.RandomApply([AddGaussianNoise(0., 0.001)], p=1.0),
    T.ToPILImage(),
    T.ToTensor(),  
])

def custom_collate(batch):
    """
    Custom collate function to convert PIL Images to tensors.
    """
    new_batch = [(transform(image), label) for image, label in batch]
    return torch.utils.data.dataloader.default_collate(new_batch)

# 設定訓練資料位置並做transform
data_dir = r'c:\software\python\simpson_aug\train\train'
dataset = CustomDataset(data_dir=data_dir, transform=transform)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in kf.split(dataset):
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)

# 定義存放訓練和驗證資料的矩陣
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

# 得到訓練的全部的角色名字種類
classes = sorted(os.listdir(data_dir))

# 訓練迴圈
for epoch in range(90):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss/len(train_loader))
        train_accuracies.append(100 * correct / total)

        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss/len(val_loader))
        val_accuracies.append(100 * correct / total)

        print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {100 * correct / total}")

        val_loss = val_loss/len(val_loader)
        scheduler.step(val_loss)

    # 每5個epochs存下其權重並測試
        if (epoch+1) % 5 == 0:
            checkpoint_path = os.path.join('c:\\software\\python\\simpson_aug', f'simpson_eff_model_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)

            # 測試模型並印出csv檔
            test_data_dir = r'c:\software\python\simpson_aug\test-final\test-final'

            class TestDataset(torch.utils.data.Dataset):
                def __init__(self, data_dir, transform=None):
                    self.data_dir = data_dir
                    self.transform = transform
                    self.image_list = [os.path.join(data_dir, img) for img in os.listdir(data_dir)]

                def __len__(self):
                    return len(self.image_list)

                def __getitem__(self, idx):
                    img_name = self.image_list[idx]
                    image = Image.open(img_name).convert('RGB')

                    if self.transform:
                        image = self.transform(image)

                    return image

            model.load_state_dict(torch.load(checkpoint_path))
            model.eval()
            model = model.to(device)

            test_dataset = TestDataset(data_dir=test_data_dir, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            predictions = []

            with torch.no_grad():
                for images in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.extend(predicted.cpu().numpy())

            test_image_list = os.listdir(test_data_dir)
            df = pd.DataFrame({'id': [int(img.split('.')[0]) for img in test_image_list], 
                            'character': [classes[i] for i in predictions]})
            df = df.sort_values(by='id')
            df.to_csv(f'c:\\software\\python\\simpson_aug\\predictions_epoch{epoch+1}.csv', index=False)


# 劃出訓練和驗證的Accuracy和loss function
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(val_accuracies, label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()


models = EfficientNet.from_name('efficientnet-b0', in_channels=3, num_classes=50) 

# 載入我epoch80的權重
weight_path = "C:/software/python/simpson_aug/simpson_eff_model_epoch80.pth"
custom_weights = torch.load(weight_path)

models._conv_stem.weight = nn.Parameter(custom_weights['_conv_stem.weight'])

# 得到一層的新權重
first_layer_weights = models._conv_stem.weight.data.cpu().numpy()

# 正規化並畫圖第1層權重
weights_min = np.min(first_layer_weights)
weights_max = np.max(first_layer_weights)
first_layer_weights = (first_layer_weights - weights_min) / (weights_max - weights_min)

n_kernels = first_layer_weights.shape[0]
n_rows = int(np.ceil(n_kernels / 8))
fig, axarr = plt.subplots(n_rows, 8, figsize=(15, 15))

for idx in range(n_kernels):
    row = idx // 8
    col = idx % 8
    ax = axarr[row, col]
    ax.imshow(np.transpose(first_layer_weights[idx], (1, 2, 0)), interpolation='nearest')
    ax.axis('off')

plt.tight_layout()
plt.show()

#得到第四層的權重
fourth_layer_weights = models._blocks[3]._depthwise_conv.weight.data.cpu().numpy()

# 正規化並畫圖第4層權重
weights_min = np.min(fourth_layer_weights)
weights_max = np.max(fourth_layer_weights)
fourth_layer_weights = (fourth_layer_weights - weights_min) / (weights_max - weights_min)

n_kernels = fourth_layer_weights.shape[0]
n_rows = int(np.ceil(n_kernels / 8))
fig, axarr = plt.subplots(n_rows, 8, figsize=(15, 15))

for idx in range(n_kernels):
    row = idx // 8
    col = idx % 8
    ax = axarr[row, col]
    ax.imshow(np.transpose(fourth_layer_weights[idx], (1, 2, 0)), interpolation='nearest')
    ax.axis('off')

plt.tight_layout()
plt.show()