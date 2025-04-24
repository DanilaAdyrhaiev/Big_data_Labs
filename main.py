import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import matplotlib.pyplot as plt
import random

# --- 1. Завантаження та підготовка даних ---
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2)

# --- 2. Архітектура мережі ---
input_size = 784
hidden_size = 128
output_size = 10

# --- Активаційні функції ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(preds, targets):
    return -np.mean(np.sum(targets * np.log(preds + 1e-9), axis=1))

def prepare_image(image_path):
    try:
        # Завантаження зображення
        img = Image.open(image_path).convert('L')  # 'L' — це режим в градаціях сірого
        img = img.resize((28, 28))  # Змінюємо розмір до 28x28

        # Перетворюємо зображення у масив numpy і масштабуємо пікселі (0-255 -> 0-1)
        img_array = np.array(img) / 255.0

        # Розв'язуємо розмірність, так щоб вхід в модель був 1x784
        img_array = img_array.flatten().reshape(1, -1)
        
        return img_array, img
    except Exception as e:
        print(f"❌ Помилка під час обробки зображення: {e}")
        return None, None

def test_image(image_path, W1, b1, W2, b2):
    # Підготовка зображення
    img_array, img = prepare_image(image_path)
    
    if img_array is None:
        return
    
    # Прямий прохід через модель
    z1 = np.dot(img_array, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    probs = softmax(z2)

    # Отримуємо передбачення (номер класу з максимальною ймовірністю)
    predicted_class = np.argmax(probs, axis=1)[0]
    
    # Відображаємо зображення та передбачення
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f"Передбачення: {predicted_class}")
    plt.axis('off')
    plt.show()
    
    # Виводимо ймовірності для всіх цифр
    print(f"Модель передбачає цифру: {predicted_class}")
    print("\nЙмовірності для кожної цифри:")
    for digit, prob in enumerate(probs[0]):
        print(f"Цифра {digit}: {prob*100:.2f}%")

def visualize_test_samples(W1, b1, W2, b2, n_samples=5):
    # Вибираємо випадкові зразки з тестової вибірки
    indices = random.sample(range(X_test.shape[0]), n_samples)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        # Отримуємо зразок
        sample = X_test[idx]
        true_label = np.argmax(y_test[idx])
        
        # Робимо передбачення
        z1 = np.dot(sample.reshape(1, -1), W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        probs = softmax(z2)
        predicted_label = np.argmax(probs, axis=1)[0]
        
        # Відображаємо зображення
        plt.subplot(1, n_samples, i+1)
        plt.imshow(sample.reshape(28, 28), cmap='gray')
        title_color = 'green' if predicted_label == true_label else 'red'
        plt.title(f"Прогноз: {predicted_label}\nФактично: {true_label}", color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- 3. Меню ---
print("1. Тренування моделі")
print("2. Тестування існуючої моделі")
print("3. Тестування власного зображення")
print("4. Візуалізація тестових прикладів")
choice = input("Оберіть пункт (1/2/3/4): ")

if choice == '1':
    # Ініціалізація ваг
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
    b2 = np.zeros((1, output_size))

    # Параметри навчання
    lr = 0.1
    epochs = 400
    batch_size = 64  # Фіксований розмір батчу

    # --- Навчання ---
    for epoch in range(epochs):
        # Перемішуємо дані перед кожною епохою
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        total_loss = 0
        batches = 0
        
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            if X_batch.shape[0] < 2:  # Пропускаємо дуже малі батчі
                continue

            # Прямий прохід
            z1 = np.dot(X_batch, W1) + b1
            a1 = relu(z1)
            z2 = np.dot(a1, W2) + b2
            probs = softmax(z2)

            # Втрата
            loss = cross_entropy(probs, y_batch)
            total_loss += loss
            batches += 1

            # Зворотний прохід
            dL_dz2 = probs - y_batch
            dW2 = np.dot(a1.T, dL_dz2) / X_batch.shape[0]
            db2 = np.mean(dL_dz2, axis=0, keepdims=True)

            dL_da1 = np.dot(dL_dz2, W2.T)
            dL_dz1 = dL_da1 * relu_derivative(z1)
            dW1 = np.dot(X_batch.T, dL_dz1) / X_batch.shape[0]
            db1 = np.mean(dL_dz1, axis=0, keepdims=True)

            # Оновлення ваг
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1
        
        # Виводимо інформацію кожні 10 епох
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / max(1, batches)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # --- Збереження ---
    np.savez('model_weights.npz', W1=W1, b1=b1, W2=W2, b2=b2)
    print("✅ Модель збережена у 'model_weights.npz'.")

    # Оцінка точності на тестовому наборі
    z1_test = np.dot(X_test, W1) + b1
    a1_test = relu(z1_test)
    z2_test = np.dot(a1_test, W2) + b2
    preds = np.argmax(softmax(z2_test), axis=1)
    true = np.argmax(y_test, axis=1)

    accuracy = np.mean(preds == true)
    print(f"🎯 Точність на тестовому наборі: {accuracy * 100:.2f}%")
    
    # Візуалізація кількох прикладів після навчання
    print("\nВізуалізація результатів на тестових прикладах:")
    visualize_test_samples(W1, b1, W2, b2, n_samples=5)

elif choice == '2':
    try:
        # --- Завантаження ---
        model = np.load('model_weights.npz')
        W1 = model['W1']
        b1 = model['b1']
        W2 = model['W2']
        b2 = model['b2']

        # --- Тестування ---
        z1_test = np.dot(X_test, W1) + b1
        a1_test = relu(z1_test)
        z2_test = np.dot(a1_test, W2) + b2
        preds = np.argmax(softmax(z2_test), axis=1)
        true = np.argmax(y_test, axis=1)

        accuracy = np.mean(preds == true)
        print(f"🎯 Точність на тестовому наборі: {accuracy * 100:.2f}%")
        
        # Показуємо матрицю помилок (які цифри модель плутає)
        confusion = np.zeros((10, 10), dtype=int)
        for i in range(len(preds)):
            confusion[true[i]][preds[i]] += 1
            
        print("\nМатриця помилок:")
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

    except FileNotFoundError:
        print("❌ Файл з моделлю не знайдено. Спочатку натренуйте модель (пункт 1).")
    except Exception as e:
        print(f"❌ Виникла помилка при завантаженні моделі: {e}")

elif choice == '3':
    # Тестування на власному зображенні
    image_path = input("Введіть шлях до зображення: ")
    try:
        # Завантажуємо модель
        model = np.load('model_weights.npz')
        W1 = model['W1']
        b1 = model['b1']
        W2 = model['W2']
        b2 = model['b2']

        # Тестуємо на власному зображенні
        test_image(image_path, W1, b1, W2, b2)

    except FileNotFoundError:
        print("❌ Файл з моделлю не знайдено. Спочатку натренуйте модель (пункт 1).")
    except Exception as e:
        print(f"❌ Виникла помилка: {e}")

elif choice == '4':
    try:
        # Завантажуємо модель
        model = np.load('model_weights.npz')
        W1 = model['W1']
        b1 = model['b1']
        W2 = model['W2']
        b2 = model['b2']
        
        # Запитуємо кількість зразків для відображення
        try:
            n_samples = int(input("Скільки зразків показати? (рекомендовано 5-10): "))
            n_samples = max(1, min(n_samples, 20))  # Обмежуємо від 1 до 20
        except ValueError:
            n_samples = 5
            print("Використовуємо значення за замовчуванням: 5 зразків")
        
        # Візуалізуємо тестові зразки
        visualize_test_samples(W1, b1, W2, b2, n_samples=n_samples)
        
    except FileNotFoundError:
        print("❌ Файл з моделлю не знайдено. Спочатку натренуйте модель (пункт 1).")
    except Exception as e:
        print(f"❌ Виникла помилка: {e}")

else:
    print("❗ Некоректний вибір. Оберіть 1, 2, 3 або 4.")