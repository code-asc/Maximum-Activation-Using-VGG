from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
from cnn import CNN



BATCH_SIZE_TRAIN = 20
BATCH_SIZE_TEST = 20
EPOCH = 40

trainset = datasets.CIFAR10(root='./data/',
                            download=False,
                            train=True,
                            transform=transforms.ToTensor())

testset = datasets.CIFAR10(root='./data/',
                    train=False, download=False, transform=transforms.ToTensor())


trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE_TRAIN)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE_TEST)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(EPOCH):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data

        # zero the parameter gradients
        model.zero_grad()

        # forward + backward + optimize
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        # print statistics
    print('epoch : ',epoch ,'loss : ', loss.item())



torch.save(model.state_dict(), './cifar_10_state.mdl')
print('Model state saved....')
torch.save(model, './cifar_10_model.mdl')
print('Model saved....')

total = 0
correct = 0


with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
