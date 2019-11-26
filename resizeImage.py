from PIL import Image, ImageDraw
import random
from matplotlib import pyplot as plt
import glob
import os
import numpy as np
import crop
#basewidth = 300
'''img = Image.open('humansPredict/017.jpg')

draw = ImageDraw.Draw(img) #Создаем инструмент для рисования.
width = img.size[0] #Определяем ширину.
height = img.size[1] #Определяем высоту.
pix = img.load() #Выгружаем значения пикселей.


wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((122,294), Image.ANTIALIAS)
img.save('humansPredict/017.jpg')'''
root_dir = 'humansTest/'

imagesPaths = glob.glob(os.path.join(root_dir, '*.jpg'))

for imagePath in imagesPaths:
    crop.resize(imagePath)