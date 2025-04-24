from DataHandler import DataHandler
from Perceptron import Perceptron
from Visualizer import Visualizer

class Menu:
    def __init__(self):
        self.data_handler = DataHandler()
        self.model = Perceptron()
        self.visualizer = Visualizer(self.model, self.data_handler)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def display_menu(self):
        print("1. Train model")
        print("2. Test existing model")
        print("3. Test custom image")
        print("4. Visualize test examples")
        print("5. Exit")
        choice = input("Select option (1/2/3/4/5): ")
        return choice

    def load_data_if_needed(self):
        if self.X_train is None:
            self.X_train, self.X_test, self.y_train, self.y_test = self.data_handler.load_and_prepare_data()

    def train_model(self):
        self.load_data_if_needed()
        self.model.train(self.X_train, self.y_train)
        self.model.save_model()
        accuracy, preds, true = self.model.evaluate(self.X_test, self.y_test)
        print(f"🎯 Test accuracy: {accuracy * 100:.2f}%")
        print("\nVisualizing results on test examples:")
        self.visualizer.visualize_test_samples(self.X_test, self.y_test, n_samples=5)

    def test_model(self):
        self.load_data_if_needed()
        
        if self.model.load_model():
            accuracy, preds, true = self.model.evaluate(self.X_test, self.y_test)
            print(f"🎯 Test accuracy: {accuracy * 100:.2f}%")
            self.visualizer.display_confusion_matrix(true, preds)

    def test_custom_image(self):
        image_path = input("Enter image path: ")
        
        if self.model.load_model():
            self.visualizer.test_image(image_path)

    def visualize_test_examples(self):
        self.load_data_if_needed()
        
        if self.model.load_model():
            try:
                n_samples = int(input("How many samples to display? (recommended 5-10): "))
                n_samples = max(1, min(n_samples, 20))  # Limit from 1 to 20
            except ValueError:
                n_samples = 5
                print("Using default value: 5 samples")
            
            self.visualizer.visualize_test_samples(self.X_test, self.y_test, n_samples=n_samples)

    def run(self):
        while True:
            choice = self.display_menu()
            if choice == '1':
                self.train_model()
            elif choice == '2':
                self.test_model()
            elif choice == '3':
                self.test_custom_image()
            elif choice == '4':
                self.visualize_test_examples()
            elif choice == '5':
                break
            else:
                print("❗ Invalid choice. Choose 1, 2, 3, 4 or 5.")