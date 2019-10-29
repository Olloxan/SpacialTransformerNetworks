
import torch

from torchvision import datasets, transforms

from Net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Dataset
if __name__ == '__main__':
     test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
            ])), batch_size=64, shuffle=True, num_workers=4)

     myNet = Net().to(device)