import numpy as np
import matplotlib.pyplot as plt
import cv2

class LetterPreprocessor():
    def __init__(self, *args, **kwargs):
        self.binValueThreshold = 45
        self.imgmaxval = 255
        self.cropPadding = 4

    def processImageShow(self, image):        
        if plt.isinteractive(): plt.ioff()

        # grayvalue picture
        image = self.makeGrayvalueImage(image)
        self.showimg(image, "input")
        
        # invert
        image = self.invertImage(image)
        self.showimg(image, "invert")

        # histogram
        thresh = self.selectThreshold(image)
        self.plotHistogram(image, thresh)
        
        #thresh = 100
        # thresholding
        image = self.thresholdImage(image, thresh)
        self.showimg(image, "thresholding")

        # opening
        image = self.openImage(image)
        self.showimg(image, "opening")

        # cropping
        image = self.cropfromBoundingBox(image)
        self.showimg(image, "croping")

        # resizing
        image = self.resizing(image)
        self.showimg(image, "resizing")
        plt.ion()
        return image

    def processImage(self, image):        
        
        # grayvalue picture
        image = self.makeGrayvalueImage(image)
               
        # invert
        image = self.invertImage(image)
       
        # histogram
        thresh = self.selectThreshold(image)
            
        #thresh = 100
        # thresholding
        image = self.thresholdImage(image, thresh)
        
        # opening
        image = self.openImage(image)
        
        # cropping
        image = self.cropfromBoundingBox(image)
        
        # resizing
        image = self.resizing(image)
        
        return image

    def showimg(self, image, title=""):
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.show()

    def plotHistogram(self, image, threshold):
        plt.hist(image.flatten(), 256, [0, 256])
        plt.title("threshold = %s" % threshold)
        plt.show()

    def makeGrayvalueImage(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def invertImage(self, image):        
        return self.imgmaxval - image

    def selectThreshold(self, image):
        hist, _ = np.histogram(image.flatten(),256, [0, 256])  
        hist[255] = 0
        threshold = 256
        for i in range(len(hist)):
            currentBinValue = hist[-i]
            if currentBinValue < self.binValueThreshold:
                threshold = 256 - i
            if currentBinValue > self.binValueThreshold:
                break
        return threshold

    def thresholdImage(self, image, threshold):
        _ , image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return image

    def openImage(self, image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def cropfromBoundingBox(self, image):
        top = self.getTop(image)
        bottom = self.getBottom(image)
        right = self.getRight(image)
        left = self.getLeft(image)
        return image[top:bottom, left:right].copy()
    
    def getTop(self, image): 
        height = image.shape[0]
        width = image.shape[1]    
        top = height
        for row in range(height):   
            for col in range(width):
                if image[row,col] > 0:
                    top = row
                    break                
            if top < height:
                break
        return max(0,top - self.cropPadding)   
    
    def getBottom(self, image):
        height = image.shape[0]
        width = image.shape[1]
        bottom = 0
        for row in reversed(range(height)):   
            for col in range(width):
                if image[row,col] > 0:
                    bottom = row
                    break                
            if bottom > 0:
                break
        return min(height, bottom + self.cropPadding)

    def getRight(self, image):
        height = image.shape[0]
        width = image.shape[1]   
        right = 0
        for col in reversed(range(width)):        
            for row in range(height):                           
                if image[row,col] > 0:
                    right = col
                    break                
            if right > 0:
                break
        return min(width, right + self.cropPadding)
        
    def getLeft(self, image):
        height = image.shape[0]
        width = image.shape[1] 
        left = width
        for col in range(width):        
            for row in range(height):                           
                if image[row,col] > 0:
                    left = col
                    break                
            if left < width:
                break
        return max(0, left - self.cropPadding)

    def resizing(self, image):
        return cv2.resize(image, (28,28),interpolation=cv2.INTER_CUBIC)
        
        
        
