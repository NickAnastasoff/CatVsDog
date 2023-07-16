import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(64 * 56 * 56, 2)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def image_transform(imagepath):
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(imagepath)
    image = transformer(image).unsqueeze(0)
    return image

def predict(imagepath, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        image = image_transform(imagepath).to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted

def main():
    imagepath = 'IMG_0659.JPG'
    model_path = 'model.pth'
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    print('Class:', predict(imagepath, model).item())
    if predict(imagepath, model).item() == 1:
        print('This is a cat.')

if __name__ == "__main__":
    main()
