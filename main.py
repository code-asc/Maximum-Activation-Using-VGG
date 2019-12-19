import torch
import numpy as np
import torchvision
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from vgg import VGG
import torch.backends.cudnn as cudnn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def imshow(img, filename=None):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    if filename is not None:
        plt.imsave(filename, np.transpose(npimg, (1, 2, 0)))

    plt.show()



PATH = './cifar_10_model.mdl'
EPOCH = 1000
ESTIMATED_TARGET = torch.tensor([0]).to(device)


X = torch.zeros((1, 3, 32, 32), requires_grad=True)
model_copy = torch.load(PATH)

loss = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    #model_copy = VGG('VGG11')
    #model_copy.load_state_dict(model.state_dict())
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=0.001, momentum=0.9)
    model_copy.zero_grad()
    optimizer.zero_grad()
    y_pred = model_copy(X)
    l = loss(y_pred, ESTIMATED_TARGET)
    l.backward()
    X.data += 0.001 * X.grad.data
    X.grad.data.zero_()




imshow(torchvision.utils.make_grid(X.detach()), 'activation')
