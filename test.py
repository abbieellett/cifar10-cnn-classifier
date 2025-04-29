import torch
import torchvision
import torchvision.transforms as transforms
from model import ClassifierCNN

device = torch.device('cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

model = ClassifierCNN().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()
