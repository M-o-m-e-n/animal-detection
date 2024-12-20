import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from tensorflow.keras.utils import load_img, img_to_array
from tqdm import tqdm
import seaborn as sns


class CustomImageClassifier:
    def __init__(self, data_dir, class_names, max_samples_per_class=None):
        self.data_dir = data_dir
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.max_samples_per_class = max_samples_per_class
        self.load_data()

    def load_data(self):
        """Load and preprocess image dataset from directory."""
        images = []
        labels = []

        for label, class_name in enumerate(tqdm(self.class_names, desc="Loading classes")):
            class_dir = os.path.join(self.data_dir, class_name)
            class_images = []
            for image_file in tqdm(os.listdir(class_dir), desc=f"Loading {class_name}", leave=False):
                if self.max_samples_per_class and len(class_images) >= self.max_samples_per_class:
                    break
                image_path = os.path.join(class_dir, image_file)
                try:
                    image = load_img(image_path, target_size=(64, 64))  # Resize to 64x64
                    image = img_to_array(image) / 255.0  # Normalize pixel values
                    class_images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")

            images.extend(class_images)

        # Convert lists to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Split into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )
        # Flatten images for classifiers like Logistic Regression, SVM
        self.X_train_flat = self.X_train.reshape(len(self.X_train), -1)
        self.X_test_flat = self.X_test.reshape(len(self.X_test), -1)
        """Load and preprocess image dataset from directory."""
        images = []
        labels = []

        for label, class_name in enumerate(tqdm(self.class_names, desc="Loading classes")):
            class_dir = os.path.join(self.data_dir, class_name)
            for image_file in tqdm(os.listdir(class_dir), desc=f"Loading {class_name}", leave=False):
                image_path = os.path.join(class_dir, image_file)
                try:
                    image = load_img(image_path, target_size=(64, 64))  # Resize to 64x64
                    image = img_to_array(image) / 255.0  # Normalize pixel values
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")

        # Convert lists to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Split into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )
        # Flatten images for classifiers like Logistic Regression, SVM
        self.X_train_flat = self.X_train.reshape(len(self.X_train), -1)
        self.X_test_flat = self.X_test.reshape(len(self.X_test), -1)

    def visualize_samples(self, num_samples=5):
        """Visualize sample images from each class."""
        plt.figure(figsize=(15, 8))
        for class_idx, class_name in enumerate(self.class_names):
            class_images = self.X_train[self.y_train == class_idx]
            for sample_idx in range(num_samples):
                plt.subplot(self.num_classes, num_samples, class_idx * num_samples + sample_idx + 1)
                plt.imshow(class_images[sample_idx])
                plt.axis('off')
                if sample_idx == 0:
                    plt.title(class_name)
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, y_true, y_pred, model_name, binary_classes=None):
        """Evaluate model performance and display metrics."""
        print(f"\n{model_name} Results:")
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("\nClassification Report:")

        if binary_classes:
            # Use the class names for the binary classes
            target_names = [self.class_names[binary_classes[0]], self.class_names[binary_classes[1]]]
        else:
            # Use all class names for multi-class classification
            target_names = self.class_names

        print(classification_report(y_true, y_pred, target_names=target_names))

        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run_svm_experiment(self):
        """Run SVM classification."""
        print("\nRunning SVM Classification...")
        svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm.fit(self.X_train_flat, self.y_train)

        predictions = []
        for x in tqdm(self.X_test_flat, desc="Predicting with SVM"):
            predictions.append(svm.predict([x])[0])
        predictions = np.array(predictions)

        self.evaluate_model(self.y_test, predictions, "SVM (RBF Kernel)")

    def run_logistic_regression(self):
        """Run binary logistic regression."""
        print("\nRunning Binary Logistic Regression...")
        log_reg = LogisticRegression(learning_rate=0.1, num_iterations=438)
        binary_classes = (0, 1)  # Example: 'butterfly' vs. 'cat'
        log_reg.fit(self.X_train_flat, self.y_train, binary_classes)

        test_mask = np.isin(self.y_test, binary_classes)
        predictions = log_reg.predict(self.X_test_flat[test_mask])
        self.evaluate_model(self.y_test[test_mask], predictions, "Logistic Regression (Binary)", binary_classes)

    def run_softmax_regression(self):
        """Run softmax regression for multi-class classification."""
        print("\nRunning Softmax Regression...")
        softmax = SoftmaxRegression(learning_rate=0.1, num_iterations=438)
        softmax.fit(self.X_train_flat, self.y_train)

        predictions = softmax.predict(self.X_test_flat)
        self.evaluate_model(self.y_test, predictions, "Softmax Regression")

    def run_knn_experiment(self, k=3):
        """Run KNN classification."""
        print(f"\nRunning KNN Classification with k={k}...")
        knn = KNNClassifier(k=k)
        knn.fit(self.X_train_flat, self.y_train)

        predictions = knn.predict(self.X_test_flat)
        self.evaluate_model(self.y_test, predictions, f"KNN (k={k})")

    def run_all_experiments(self):
        """Run all experiments."""
        """self.run_logistic_regression()
        self.run_softmax_regression()
        self.run_svm_experiment()"""
        self.run_knn_experiment(k=3)  # Example with k=3


class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=438):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, binary_classes=(0, 1)):
        # Filter data for binary classification
        mask = np.isin(y, binary_classes)
        X = X[mask]
        y = y[mask]
        y = (y == binary_classes[1]).astype(int)

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.num_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            # Backward pass
            dz = predictions - y
            dw = np.dot(X.T, dz) / len(y)
            db = np.sum(dz) / len(y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return (predictions >= 0.5).astype(int)


class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, num_iterations=438):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def softmax(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, y):
        num_classes = len(np.unique(y))
        self.weights = np.zeros((X.shape[1], num_classes))
        self.bias = np.zeros(num_classes)

        # One-hot encode labels
        y_onehot = np.zeros((len(y), num_classes))
        y_onehot[np.arange(len(y)), y] = 1

        for _ in range(self.num_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.softmax(z)

            # Backward pass
            error = predictions - y_onehot
            dw = np.dot(X.T, error) / len(y)
            db = np.sum(error, axis=0) / len(y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return np.argmax(self.softmax(z), axis=1)


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in tqdm(X, desc="Predicting with KNN"):
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            prediction = np.bincount(k_nearest_labels).argmax()
            predictions.append(prediction)
        return np.array(predictions)


if __name__ == "__main__":
    # Initialize the classifier
    DATA_DIR = "/kaggle/input/animals10/raw-img"
    CLASS_NAMES = ['cane', 'cavallo', 'elefante', 'farfalla', 'gatto',
                   'mucca', 'pecora', 'ragno', 'scoiattolo']
    classifier = CustomImageClassifier(DATA_DIR, CLASS_NAMES)

    # Visualize sample images
    classifier.visualize_samples()

    # Run experiments
    classifier.run_all_experiments()
