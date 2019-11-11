import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

from Net import Net_EMNIST

#plt.ion()

device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    
    img = cv2.imread("TestData/A2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap="gray")
    plt.show()

    test =    cv2.mean(img)   
    img = 255 - img    

    plt.imshow(img, cmap="gray")
    plt.show()

    hist, bins = np.histogram(img.flatten(),256, [0, 256])   
    
    for i in range(len(hist)):
        x = hist[-i]
        if x < 50:
            threshold = 256 - i
        if x > 50:
            break

    plt.hist(img.flatten(),256,[0,256])
    plt.show()

    
    _ , img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    plt.imshow(img, cmap="gray")
    plt.show()

    img = cv2.resize(img, (28,28),interpolation=cv2.INTER_CUBIC)

    plt.imshow(img, cmap="gray")
    plt.show()

    myTransforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))])
    img = myTransforms(img)
    img = img.unsqueeze(0)

    myNet = Net_EMNIST().to(device)
    pretrained_dict = torch.load("Models/EMNIST_Spacial", map_location='cpu')
    myNet.load_state_dict(pretrained_dict)

    output = myNet(img)
    prediction = output.max(1, keepdim=True)[1]
    x=5