import cv2
import numpy as np
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
    nameOfNewDirFile = "test_humans_not_ready//"+nameOfFile[-7:]
    im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90).save(nameOfNewDirFile)

    img2 = cv2.imread(nameOfNewDirFile)
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
    nameOfNewDirFile = "test_humans_ready//" + nameOfFile[-7:]
    cv2.imwrite(nameOfNewDirFile, new_img)

    basewidth = 300
    target_width = 122
    target_height = 294
    img = Image.open(nameOfNewDirFile)

    draw = ImageDraw.Draw(img)  # Создаем инструмент для рисования.
    width = img.size[0]  # Определяем ширину.
    height = img.size[1]  # Определяем высоту.
    pix = img.load()  # Выгружаем значения пикселей.

    wpercent = (basewidth / float(img.size[0]))
    # Масштабируем до нужного размера
    img = img.resize((target_width, target_height), Image.ANTIALIAS)
    # Сохраняем
    nameOfNewDirFile = "test_only_humans_ready//" + nameOfFile[-7:]
    img.save(nameOfNewDirFile)

