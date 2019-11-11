import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_MNIST(nn.Module):
    def __init__(self):
        super(Net_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

        # Spatial transformer localization network
        self.loacalization = nn.Sequential(nn.Conv2d(1,8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8,10,kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True))

        # Regressor for the 3*2 affine matrix
        self.fc_loc = nn.Sequential(nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32,3 * 2))

        # initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))

    def stn(self, x):
        xs = self.loacalization(x)
        xs = xs.view(-1,10*3*3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net_EMNIST(nn.Module):
    def __init__(self):
        super(Net_EMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,512)
        self.fc2 = nn.Linear(512,100)
        self.fc3 = nn.Linear(100,47)

        # Spatial transformer localization network
        self.loacalization = nn.Sequential(nn.Conv2d(1,8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8,10,kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True))

        # Regressor for the 3*2 affine matrix
        self.fc_loc = nn.Sequential(nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32,3 * 2))

        # initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))

    def stn(self, x):
        xs = self.loacalization(x)
        xs = xs.view(-1,10*3*3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Net_EMNIST_NEU(nn.Module):
    def __init__(self):
        super(Net_EMNIST_NEU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,10,kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10,20, kernel_size=5),
            nn.Dropout2d(),                 # default p=0.5
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
            )

        self.fc = nn.Sequential(
            nn.Linear(320,512),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(512,100),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(100,47)
            )
      

        # Spatial transformer localization network
        self.loacalization = nn.Sequential(nn.Conv2d(1,8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8,10,kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True))

        # Regressor for the 3*2 affine matrix
        self.fc_loc = nn.Sequential(nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32,3 * 2))

        # initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))

    def stn(self, x):
        xs = self.loacalization(x)
        xs = xs.view(-1,10*3*3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = self.conv(x)
        x = x.view(-1,320)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)