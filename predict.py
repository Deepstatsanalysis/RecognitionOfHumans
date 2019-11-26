from keras.models import load_model
import glob
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

def resize(nameOfFile):
    # Открываем изображение
    img = cv2.imread(nameOfFile)
    # переводим в формат hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # бинаризуем изображение по черному цвету
    mask = cv2.inRange(hsv, (0, 0, 0), (0, 0, 0))
    # находим пиксели черного цвета по x и y
    y, x = np.where(mask != 0)

    # Находим координаты верхнего левого угла по y
    y_start = y[0]
    # Находим координаты нижнего правого угла по y
    y_end = y[-1]

    # Переворачиваем, отображаем по вертикали (для поиска углов по х)
    im = Image.open(nameOfFile)
    im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90).save("_"+nameOfFile)

    img2 = cv2.imread("_"+nameOfFile)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    mask2 = cv2.inRange(hsv2, (0, 0, 0), (0, 0, 0))
    y, x = np.where(mask2 != 0)

    # Находим координаты верхнего левого угла по x
    x_start = y[0]
    # Находим координаты нижнего правого угла по x
    x_end = y[-1]

    print(x_start)
    print(y_start)
    print(x_end)
    print(y_end)

    # Обрезаем изображение по прямоугольнику
    new_img = img[y_start:y_end, x_start:x_end]
    plt.imshow(new_img)
    # Сохраняем
    cv2.imwrite("only"+nameOfFile, new_img)

    basewidth = 300
    target_width = 122
    target_height = 294
    img = Image.open("only"+nameOfFile)

    draw = ImageDraw.Draw(img)  # Создаем инструмент для рисования.
    width = img.size[0]  # Определяем ширину.
    height = img.size[1]  # Определяем высоту.
    pix = img.load()  # Выгружаем значения пикселей.

    wpercent = (basewidth / float(img.size[0]))
    # Масштабируем до нужного размера
    img = img.resize((target_width, target_height), Image.ANTIALIAS)
    # Сохраняем
    img.save("only"+nameOfFile)

model = load_model('models/modelBestKeks.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_test = []

root_dir = 'humansPredict/'

imagesPaths = glob.glob(os.path.join(root_dir, '*.jpg'))

for imagePath in imagesPaths:
    resize(imagePath)

root_dir = 'onlyhumansPredict/'

imagesPaths = glob.glob(os.path.join(root_dir, '*.jpg'))

for imagePath in imagesPaths:
    image = cv2.imread(imagePath)
    X_test.append(image)

X_test = np.asarray(X_test)

print(X_test.shape)

X_test = np.array(X_test, dtype="float32") / 255.0
prediction = model.predict_classes(X_test)

count = 0
for i in prediction:
    count += 1
    if i == 0:
        print('{}: вероятно худой человечек'.format(count))
    else:
        print('{}: вероятно толстый человечек'.format(count))