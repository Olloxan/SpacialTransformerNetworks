import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

from Net import Net_EMNIST

plt.ion()

device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

def showimg(image, title):
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    # input
    img = cv2.imread("TestData/t_1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    showimg(img, "input")
    
    # invert
    img = 255 - img    
    showimg(img, "inverting")

    # histogram
    hist, bins = np.histogram(img.flatten(),256, [0, 256])   
    
    for i in range(len(hist)):
        x = hist[-i]
        if x < 45:
            threshold = 256 - i
        if x > 45:
            break
        
    plt.hist(img.flatten(),256,[0,256])
    plt.title("threshold = %s" % threshold)
    plt.show()

    # thresholding
    _ , img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    showimg(img, "thresholding")

    # opening
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    showimg(img, "opening")
    
    # bounding box
   
    height = img.shape[0]
    width = img.shape[1]
    print("height = %s, width = %s" % (height, width))
    padding = 4

    # left
    debugimg = img.copy()
    left = width
    for col in range(width):        
        for row in range(height):                           
            if img[row,col] > 0:
                left = col
                break
            debugimg[row,col] = 128
        if left < width:
            break
    left = max(0, left - padding)
    print("left = %s" % left)

    #plt.imshow(debugimg, cmap="gray")
    #plt.show()
   
    # right
    debugimg = img.copy()
    right = 0
    for col in reversed(range(width)):        
        for row in range(height):                           
            if img[row,col] > 0:
                right = col
                break
            debugimg[row,col] = 128
        if right > 0:
            break
    right = min(width, right + padding)
    print("right = %s" % right)

    #plt.imshow(debugimg, cmap="gray")
    #plt.show()
    
    # top
    debugimg = img.copy()
    top = height
    for row in range(height):   
        for col in range(width):
            if img[row,col] > 0:
                top = row
                break
            debugimg[row,col] = 128
        if top < height:
            break
    top = max(0,top - padding)
    print("top = %s" % top)

    #plt.imshow(debugimg, cmap="gray")
    #plt.show()

    # bottom
    debugimg = img.copy()
    bottom = 0
    for row in reversed(range(height)):   
        for col in range(width):
            if img[row,col] > 0:
                bottom = row
                break
            debugimg[row,col] = 128
        if bottom > 0:
            break
    bottom = min(height,bottom + padding)
    print("bottom = %s" % bottom)

    #plt.imshow(debugimg, cmap="gray")
    #plt.show()

    # crop
    img = img[top:bottom, left:right].copy()
    showimg(img, "cropping")

    # resize
    img = cv2.resize(img, (28,28),interpolation=cv2.INTER_CUBIC)

    showimg(img, "resizing")

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
    print(prediction.item())
    x=5