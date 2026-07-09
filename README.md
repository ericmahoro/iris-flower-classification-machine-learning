# Iris Flower Species Classification Using Machine Learning

## Project Overview

This project implements multiple machine learning classification algorithms to predict Iris flower species based on physical measurements.

The goal was to compare different classification approaches and evaluate their performance using accuracy, confusion matrix, precision, recall, and F1-score.

## Dataset

The project uses the Iris dataset from Scikit-learn.

Dataset characteristics:

- Samples: 150 flowers
- Features: 4
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- Classes:
  - Setosa
  - Versicolor
  - Virginica

Dataset distribution:

- Setosa: 50 samples
- Versicolor: 50 samples
- Virginica: 50 samples

## Machine Learning Models

The following classification models were implemented:

### Rule-Based and Probabilistic Models

- Decision Tree
- Rule-Based Classifier
- Gaussian Naive Bayes

### Distance and Optimization-Based Models

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

## Model Evaluation

Models were evaluated using:

- Confusion Matrix
- Precision
- Recall
- F1-score
- Accuracy

## Results

Five models achieved perfect classification performance:

| Model | Accuracy |
|---|---|
| Decision Tree | 100% |
| Naive Bayes | 100% |
| Logistic Regression | 100% |
| KNN | 100% |
| SVM | 100% |

The Rule-Based classifier achieved:

Accuracy: 93.3%

## Conclusion

The Iris dataset contains clearly separable flower classes, especially because petal measurements provide strong distinguishing characteristics.

The best-performing models were Decision Tree, Naive Bayes, Logistic Regression, KNN, and SVM, all achieving 100% accuracy on the test dataset.

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Machine Learning Classification Algorithms

## How to Run

Clone the repository:

```bash
git clone https://github.com/ericmahoro/iris-flower-classification-machine-learning.git

Install dependencies:

pip install -r requirements.txt

Run the project:

python iris_classification.py



