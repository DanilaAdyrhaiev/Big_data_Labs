import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

def display_ten_images():
    for i in range(10):
        img = random.randint(0, 70000)
        image = X[img].reshape(28, 28)
        chars = " .:-=+*#%@"
        for row in image:
            for pixel in row:
                brightness = int(pixel / 255 * (len(chars) - 1))
                print(chars[brightness], end='')
            print()

def get_image():
    img = random.randint(0, 70000)
    image = X[img].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {y[img]}")
    plt.axis('off')
    plt.show()

display_ten_images()