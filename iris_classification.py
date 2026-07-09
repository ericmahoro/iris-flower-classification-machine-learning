# Import libraries
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score



# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add target column
df["species"] = iris.target

# Display first 5 rows
print(df.head())

# Dataset characteristics
print("\nDataset Shape:")
print(df.shape)

print("\nFeature Names:")
print(iris.feature_names)

print("\nClass Names:")
print(iris.target_names)

print("\nClass Distribution:")
print(df["species"].value_counts())

# Split dataset

X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining data size:")
print(X_train.shape)


print("\nTesting data size:")
print(X_test.shape)

# -------------------------------
# Model 1: Decision Tree
# -------------------------------

decision_tree = DecisionTreeClassifier(
    random_state=42
)

# Train the model
decision_tree.fit(X_train, y_train)

# Make predictions
dt_predictions = decision_tree.predict(X_test)


print("\nDecision Tree Predictions:")
print(dt_predictions)

# -------------------------------
# Model 2: Rule-Based Classifier
# -------------------------------

def rule_based_classifier(data):
    predictions = []

    for sample in data:
        petal_length = sample[2]
        petal_width = sample[3]

        if petal_length <= 2.5:
            predictions.append(0)   # Setosa

        elif petal_width <= 1.8:
            predictions.append(1)   # Versicolor

        else:
            predictions.append(2)   # Virginica

    return predictions


# Apply rule-based classifier
rule_predictions = rule_based_classifier(X_test)


print("\nRule-Based Classifier Predictions:")
print(rule_predictions)

# -------------------------------
# Model 3: Naïve Bayes
# -------------------------------

naive_bayes = GaussianNB()

# Train the model
naive_bayes.fit(X_train, y_train)

# Make predictions
nb_predictions = naive_bayes.predict(X_test)


print("\nNaïve Bayes Predictions:")
print(nb_predictions)

# -------------------------------
# Model 4: Logistic Regression
# -------------------------------

logistic_model = LogisticRegression(
    max_iter=200,
    random_state=42
)

# Train the model
logistic_model.fit(X_train, y_train)

# Make predictions
lr_predictions = logistic_model.predict(X_test)


print("\nLogistic Regression Predictions:")
print(lr_predictions)

# -------------------------------
# Model 5: K-Nearest Neighbors
# -------------------------------

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions
knn_predictions = knn_model.predict(X_test)


print("\nKNN Predictions:")
print(knn_predictions)

# -------------------------------
# Model 6: Support Vector Machine
# -------------------------------

svm_model = SVC(
    kernel="linear",
    random_state=42
)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions
svm_predictions = svm_model.predict(X_test)


print("\nSVM Predictions:")
print(svm_predictions)

print("\nModel Accuracy:")

print("Decision Tree Accuracy:",
      accuracy_score(y_test, dt_predictions))

print("Rule-Based Accuracy:",
      accuracy_score(y_test, rule_predictions))

print("Naive Bayes Accuracy:",
      accuracy_score(y_test, nb_predictions))

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_predictions))

print("KNN Accuracy:",
      accuracy_score(y_test, knn_predictions))

print("SVM Accuracy:",
      accuracy_score(y_test, svm_predictions))

models = {
    "Decision Tree": dt_predictions,
    "Rule-Based": rule_predictions,
    "Naive Bayes": nb_predictions,
    "Logistic Regression": lr_predictions,
    "KNN": knn_predictions,
    "SVM": svm_predictions
}


for model_name, predictions in models.items():

    print("\n==============================")
    print(model_name)
    print("==============================")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    ))

print("\n==============================")
print("MODEL PERFORMANCE COMPARISON")
print("==============================")


models = {
    "Decision Tree": dt_predictions,
    "Rule-Based": rule_predictions,
    "Naive Bayes": nb_predictions,
    "Logistic Regression": lr_predictions,
    "KNN": knn_predictions,
    "SVM": svm_predictions
}


for name, prediction in models.items():

    precision = precision_score(y_test, prediction, average="weighted")
    recall = recall_score(y_test, prediction, average="weighted")
    f1 = f1_score(y_test, prediction, average="weighted")

    print("\n", name)
    print("----------------")
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1-score:", round(f1, 3))

