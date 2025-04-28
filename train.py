import torch
import torchvision
import torchvision.transforms as transforms
from model import ClassifierCNN

device = torch.device('cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#*********** batch size can be changed ***********
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainlaoder = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# model, loss and optimiser
model = ClassifierCNN().to(device)
loss = torch.nn.CrossEntropyLoss()
#*********** lr can be changed ***********
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
