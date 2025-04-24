from sklearn.datasets import fetch_openml
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

class DataHandler:
    def __init__(self):
        self.encoder = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_prepare_data(self):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = X / 255.0  # Нормалізація

        # Ван-хот
        self.encoder = OneHotEncoder(sparse_output=False)
        y_onehot = self.encoder.fit_transform(y.reshape(-1, 1))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_onehot, test_size=0.2
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def prepare_image(self, image_path):
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img) / 255.0
            img_array = img_array.flatten().reshape(1, -1)
            
            return img_array, img
        except Exception as e:
            print(f"Помилка обробки зображення: {e}")
            return None, None
