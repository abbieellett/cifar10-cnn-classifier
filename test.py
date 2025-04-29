import torch
import torchvision
import torchvision.transforms as transforms
from model import ClassifierCNN

if __name__ == '__main__':
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

    correct = 0
    total = 0
    # not tracking gradients
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * (correct/total):.2f}%')