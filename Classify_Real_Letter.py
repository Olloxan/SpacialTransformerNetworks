import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

from Net import Net_EMNIST_3
from Utilitiy.LetterPreprocessor import LetterPreprocessor

device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = LetterPreprocessor()

def showimg(image, title):
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    # input
    img = cv2.imread("TestData/1_1.jpg")
    img = processor.processImageShow(img)
    #img = processor.processImage(img)

    myTransforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))])
    img = myTransforms(img)
    img = img.unsqueeze(0)

    myNet = Net_EMNIST_3().to(device)
    pretrained_dict = torch.load("Models/EMNIST_Spacial_3", map_location='cpu')
    myNet.load_state_dict(pretrained_dict)

    output = myNet(img)
    test = np.exp(output.detach().squeeze(0).numpy())
    #plt.plot(test)
    y_pos = np.arange(len(test))
    plt.bar(y_pos, test, align='center')    
    plt.xticks(y_pos)
    plt.show()
    prediction = output.max(1, keepdim=True)[1]
    print(prediction.item())
    x=5