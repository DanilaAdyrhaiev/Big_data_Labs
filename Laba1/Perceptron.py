import numpy as np

class Perceptron:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def cross_entropy(self, preds, targets):
        return -np.mean(np.sum(targets * np.log(preds + 1e-9), axis=1))

    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        probs = self.softmax(z2)
        return z1, a1, z2, probs

    def train(self, X_train, y_train, lr=0.1, epochs=400, batch_size=64):
        for epoch in range(epochs):
            # Shuffle data before each epoch
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            total_loss = 0
            batches = 0
            
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                if X_batch.shape[0] < 2:  # Skip very small batches
                    continue

                # Forward pass
                z1, a1, z2, probs = self.forward(X_batch)

                # Loss
                loss = self.cross_entropy(probs, y_batch)
                total_loss += loss
                batches += 1

                # Backward pass
                dL_dz2 = probs - y_batch
                dW2 = np.dot(a1.T, dL_dz2) / X_batch.shape[0]
                db2 = np.mean(dL_dz2, axis=0, keepdims=True)

                dL_da1 = np.dot(dL_dz2, self.W2.T)
                dL_dz1 = dL_da1 * self.relu_derivative(z1)
                dW1 = np.dot(X_batch.T, dL_dz1) / X_batch.shape[0]
                db1 = np.mean(dL_dz1, axis=0, keepdims=True)

                # Update weights
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
            
            # Output information every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / max(1, batches)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        _, _, _, probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def get_probabilities(self, X):
        _, _, _, probs = self.forward(X)
        return probs

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        true = np.argmax(y_test, axis=1)
        accuracy = np.mean(preds == true)
        return accuracy, preds, true

    def save_model(self, filename='model_weights.npz'):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"✅ Model saved to '{filename}'.")

    def load_model(self, filename='model_weights.npz'):
        try:
            model = np.load(filename)
            self.W1 = model['W1']
            self.b1 = model['b1']
            self.W2 = model['W2']
            self.b2 = model['b2']
            return True
        except FileNotFoundError:
            print("❌ Model file not found. Train the model first (option 1).")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False