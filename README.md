# Custom Image Classifier

This project implements a machine learning-based image classification system using Python. It includes preprocessing of image datasets, visualization of sample images, and experiments with different classification models such as Support Vector Machines (SVM), Logistic Regression, and Softmax Regression.

## Features
- **Dataset Handling**: Load and preprocess images from a directory-based dataset.
- **Visualization**: Display sample images for each class.
- **Classification Models**:
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Softmax Regression
- **Evaluation**: Accuracy, classification reports, and confusion matrix visualizations.

## Dataset
The dataset should be organized into a directory with subfolders for each class. Each subfolder should contain the images for that class.

### Example Directory Structure
```
archive/raw-img/
├── butterfly/
├── cat/
├── chicken/
├── dog/
├── elephant/
├── horse/
├── sheep/
├── spider/
└── squirrel/
```

## Prerequisites
- Python 3.7 or higher
- Libraries:
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `tensorflow`
  - `scikit-learn`
  - `tqdm`

Install the dependencies using pip:
```bash
pip install numpy matplotlib seaborn tensorflow scikit-learn tqdm
```

## Usage

### 1. Initialize the Classifier
Modify the `DATA_DIR` and `CLASS_NAMES` in the script to match your dataset:
```python
DATA_DIR = "archive/raw-img"
CLASS_NAMES = ['butterfly', 'cat', 'chicken', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
classifier = CustomImageClassifier(DATA_DIR, CLASS_NAMES, max_samples_per_class=100)
```

### 2. Visualize Sample Images
To view a few samples from each class:
```python
classifier.visualize_samples()
```

### 3. Run Classification Experiments
Run all classification experiments (Logistic Regression, Softmax Regression, and SVM):
```python
classifier.run_all_experiments()
```

### 4. Model-Specific Experiments
Run specific experiments:
- **SVM**:
  ```python
  classifier.run_svm_experiment()
  ```
- **Logistic Regression**:
  ```python
  classifier.run_logistic_regression()
  ```
- **Softmax Regression**:
  ```python
  classifier.run_softmax_regression()
  ```

## Class Implementation Details

### `CustomImageClassifier`
- **Methods**:
  - `load_data()`: Loads and preprocesses the dataset.
  - `visualize_samples(num_samples)`: Visualizes sample images from each class.
  - `evaluate_model(y_true, y_pred, model_name, binary_classes)`: Evaluates model performance.
  - `run_svm_experiment()`: Runs an SVM classification experiment.
  - `run_logistic_regression()`: Runs a binary logistic regression experiment.
  - `run_softmax_regression()`: Runs a multi-class softmax regression experiment.
  - `run_all_experiments()`: Runs all supported experiments.

### `LogisticRegression`
Implements binary logistic regression with gradient descent optimization.

### `SoftmaxRegression`
Implements multi-class classification using softmax regression with gradient descent.

## Results
Results include:
- **Accuracy**: Displayed for each model.
- **Classification Report**: Includes precision, recall, and F1-score.
- **Confusion Matrix**: Visualized as a heatmap.

## Example Output
- **Confusion Matrix Heatmap**:
  ![Confusion Matrix Heatmap Example](example_heatmap.png)

- **Sample Visualization**:
  ![Sample Visualization Example](example_samples.png)

## Contributing
Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the MIT License.
