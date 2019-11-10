from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms

import numpy as np

from Net import Net_EMNIST

plt.ion()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target =data.to(device), target.to(device)

        optimizer.zero_grad()
        output = myNet(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%]\tLoss:{:.06f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    with torch.no_grad():
        myNet.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = myNet(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = myNet.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.



# Training Dataset
if __name__ == '__main__':
    cudnnVersion = torch.backends.cudnn.version()

    transformations = transforms.Compose([
        transforms.RandomRotation([90,90]),
        transforms.RandomVerticalFlip(1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(root='.', split='balanced', train=True, download=False, 
                       transform=transformations), batch_size=32, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(root='.', split='balanced', train=False, 
                        transform=transformations), batch_size=32, shuffle=True, num_workers=4)

    myNet = Net_EMNIST().to(device)

    optimizer = optim.SGD(myNet.parameters(), lr=0.01)

    for epoch in range(1):#, 20 + 1):
        #train(epoch)
        test()

    #state_dict = myNet.state_dict()
    #torch.save(state_dict, "Models/EMNIST_Spacial")
    print("Network Saved")
    # Visualize the STN transformation on some input batch
    visualize_stn()

    plt.ioff()
    plt.show()
