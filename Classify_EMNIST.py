import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets, transforms

from Net import Net_EMNIST_1

device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        myNet.eval()
        dataTuple = next(iter(test_loader)) 
        data = dataTuple[0].to(device)
        labels = dataTuple[1]

        input_tensor = data.cpu()
        transformed_input_tensor = myNet.stn(data).cpu()

        output = myNet(data)
        
        test = output[0].cpu().numpy()

        predictions = output.max(1, keepdim=True)[1]
        
        for tuple in zip(labels, predictions):
            print(tuple[0].item(), tuple[1].item())
        #print(prediction.item())
        
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

# Training Dataset
if __name__ == '__main__':
     test_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(root='.', split='balanced', train=False, transform=transforms.Compose([
            transforms.RandomRotation([90,90]),
            transforms.RandomVerticalFlip(1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
            ])), batch_size=10, shuffle=True, num_workers=4)

     myNet = Net_EMNIST_1()
     pretrained_dict = torch.load("Models/EMNIST_Spacial_1", map_location='cpu')
     myNet.load_state_dict(pretrained_dict)
     
     visualize_stn()
     plt.show()
