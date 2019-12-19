from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
from vgg import VGG
import torch.backends.cudnn as cudnn


device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 100
EPOCH = 150

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

print('Preprocessing dataset started....')


trainset = datasets.CIFAR10(root='codeasc/datasets/cifar10/1/',
                            download=True,
                            train=True,
                            transform=transform_train)


testset = datasets.CIFAR10(root='codeasc/datasets/cifar10/1/',
                    train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE_TRAIN)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE_TEST)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('Preprocessing dataset finished....')
print('Model training started....')

model = VGG('VGG11')

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


for epoch in range(EPOCH):  # loop over the dataset multiple times

    print('current epoch : ', epoch)

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = inputs.to(device), targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

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
    for batch_idx, (inputs, targets) in enumerate(testloader):
        images, labels = inputs.to(device), targets.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
