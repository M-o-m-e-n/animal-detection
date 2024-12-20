import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
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

    def apply_custom_processing(self, image):
        """Apply custom image processing techniques."""
        # Convert to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply histogram equalization
        equalized = cv2.equalizeHist(grayscale)

        # Edge detection
        edges = cv2.Canny(equalized, 100, 200)

        return edges

    def run_neural_network(self):
        """Run a simple feed-forward neural network."""
        print("\nRunning Feed-Forward Neural Network...")
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train_flat.shape[1],)),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.X_train_flat, self.y_train, epochs=10, batch_size=32, validation_split=0.2)
        predictions = np.argmax(model.predict(self.X_test_flat), axis=1)
        self.evaluate_model(self.y_test, predictions, "Feed-Forward Neural Network")

    def run_cnn(self):
        """Run a Convolutional Neural Network (CNN)."""
        print("\nRunning Convolutional Neural Network...")
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.X_train.shape[1:]),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, validation_split=0.2)
        predictions = np.argmax(model.predict(self.X_test), axis=1)
        self.evaluate_model(self.y_test, predictions, "Convolutional Neural Network")

    def run_all_experiments(self):
        """Run all experiments."""
        self.run_logistic_regression()
        self.run_softmax_regression()
        self.run_svm_experiment()
        self.run_knn_experiment(k=3)
        self.run_neural_network()
        self.run_cnn()


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
