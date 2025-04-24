import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

class Visualizer:
    def __init__(self, model, data_handler):
        self.model = model
        self.data_handler = data_handler

    def visualize_test_samples(self, X_test, y_test, n_samples=5):
        indices = random.sample(range(X_test.shape[0]), n_samples)
        
        plt.figure(figsize=(15, 3))
        for i, idx in enumerate(indices):
            sample = X_test[idx]
            true_label = np.argmax(y_test[idx])
            predicted_label = self.model.predict(sample.reshape(1, -1))[0]
            plt.subplot(1, n_samples, i+1)
            plt.imshow(sample.reshape(28, 28), cmap='gray')
            title_color = 'green' if predicted_label == true_label else 'red'
            plt.title(f"Pred: {predicted_label}\nActual: {true_label}", color=title_color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def test_image(self, image_path):
        img_array, img = self.data_handler.prepare_image(image_path)
        if img_array is None:
            return
        probs = self.model.get_probabilities(img_array)
        predicted_class = np.argmax(probs, axis=1)[0]
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.title(f"Prediction: {predicted_class}")
        plt.axis('off')
        plt.show()
        print(f"Model predicts digit: {predicted_class}")
        print("\nProbabilities for each digit:")
        for digit, prob in enumerate(probs[0]):
            print(f"Digit {digit}: {prob*100:.2f}%")

    def display_confusion_matrix(self, true, preds):
        confusion = np.zeros((10, 10), dtype=int)
        for i in range(len(preds)):
            confusion[true[i]][preds[i]] += 1
            
        print("\nConfusion Matrix:")
        print("  ", end="")
        for i in range(10):
            print(f"  {i}  ", end="")
        print("\n")
        
        for i in range(10):
            print(f"{i} ", end="")
            for j in range(10):
                if i == j:
                    print(f"[{confusion[i][j]:3d}]", end=" ")
                else:
                    print(f" {confusion[i][j]:3d} ", end=" ")
            print()

        