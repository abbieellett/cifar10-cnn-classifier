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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# model, loss and optimiser
model = ClassifierCNN().to(device)
loss_function = torch.nn.CrossEntropyLoss()
#*********** lr can be changed ***********
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

# training loop
epoch_num = 10
for epoch in range(epoch_num):
    batch_losses = 0.0
    correct_class = 0 # number of images classified correctly
    total = 0 # total images seen so far
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimiser.zero_grad() # clear old gradient from last batch
        outputs = model(inputs) # run through model
        loss = loss_function(outputs, labels)
        loss.backward() # calculate gradients of loss
        optimiser.step() # update model parameters

        batch_losses += loss.item()
        _, predicted = torch.max(outputs, 1) # discard max values
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Epoch {epoch+1}/{epoch_num}, Loss: {batch_losses/len(trainloader):.3f}, Accuracy: {100*(correct_class/total):.2f}%')

# save model
torch.save(model.state_dict(), 'model.pth')
print('Training finished, new model saved')
