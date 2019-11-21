import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision
#import torchsummary
from torchvision import transforms
import torch

from Net import Net_EMNIST_3
from Utilitiy.LetterPreprocessor import LetterPreprocessor

device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = LetterPreprocessor()

halloweltpath = "TestData/HalloWelt/"
mlisttollpath = "TestData/DasMLMeetupisttoll/"

path = mlisttollpath

letterdict =	{
  0: "0",
  1: "1",
  2: "2",
  3: "3",
  4: "4",
  5: "5",
  6: "6",
  7: "7",
  8: "8",
  9: "9",
  10: "A",
  11: "B",
  12: "C",
  13: "D",
  14: "E",
  15: "F",
  16: "G",
  17: "H",
  18: "I",
  19: "J",
  20: "K",
  21: "L",
  22: "M",
  23: "N",
  24: "O",
  25: "P",
  26: "Q",
  27: "R",
  28: "S",
  29: "T",
  30: "U",
  31: "V",
  32: "W",
  33: "X",
  34: "Y",
  35: "Z",
  36: "a",
  37: "b",
  38: "d",
  39: "e",
  40: "f",
  41: "g",
  42: "h",
  43: "n",
  44: "q",
  45: "r",
  46: "t"
}

def showimg(image, title):
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.show()

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

if __name__ == '__main__':
    myNet = Net_EMNIST_3().to(device)
    pretrained_dict = torch.load("Models/EMNIST_Spacial_3", map_location='cpu')
    myNet.load_state_dict(pretrained_dict)

    myTransforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))])
    
    ###############################
    # input
   
   
    words = []
    folders = os.listdir(path)
    for folderName in folders:  
        currentfolderpath = path + folderName + "/"
        files = os.listdir(currentfolderpath)
        
        images = torch.FloatTensor()
        for i in range(len(files)):            
            img = cv2.imread(currentfolderpath + "%s.jpg" % (i+1))
            #img = processor.processImageShow(img)
            img = processor.processImage(img)    
            img = myTransforms(img).unsqueeze(0)    
            images = torch.cat((images,img), 0)
        words.append(images)
  

    with torch.no_grad():
        # alle mit einem mal durchjagen
        myNet.eval()        
      
        transformcat = torch.cat(words, 0)
        transformed_input_tensor = myNet.stn(transformcat).cpu() # transformierter Input                                     
        
        in_grid = convert_image_np(
            torchvision.utils.make_grid(transformcat))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))
                        
        # Plot the results side-by-side
        _, ax = plt.subplots(2, 1)
        ax[0].imshow(in_grid)
        ax[0].set_title('Dataset Images')

        ax[1].imshow(out_grid)
        ax[1].set_title('Transformed Images')
        
        outputstring = ""
        for word in words:            
            output = myNet(word)
            prediction = output.max(1, keepdim=True)[1].squeeze(1).numpy()
            
            for sign in prediction:
                outputstring += letterdict[sign]
            outputstring += " "
        
        plt.figure()
        plt.text(0, 0.5, outputstring, size=50)
        plt.axis('off')
        plt.show()
    
    ##############################


    #test = np.exp(output.detach().squeeze(0).numpy())    
    #y_pos = np.arange(len(test))
    #plt.bar(y_pos, test, align='center')    
    #plt.xticks(y_pos)
    #plt.show()
    