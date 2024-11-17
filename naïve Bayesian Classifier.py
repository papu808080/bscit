from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Example dataset
documents = [
    "This is a positive document",
    "Negative sentiment in this text",
    "A very positive review",
    "Review with a negative tone",
    "Neutral document here"
]


labels = ["positive", "negative", "positive", "negative", "neutral"]


# Create a vectorizer to convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)


# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


# Create and train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# Predict the classes for the test data
y_pred = classifier.predict(X_test)


# Calculate and print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


# Print actual and predicted classes for the test data
print("Test Data:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")




print(f"\nAccuracy: {accuracy:.2f}")
print(report)


# Naive Bayes classifier for Numerical data
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report


# Load Iris dataset
data = load_iris()
X = data.data
y = data.target


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialize and train Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)


# Predict on test set
y_pred = model.predict(X_test)


# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')


print("Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)


# Predict on new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example input (sepal length, sepal width, petal length, petal width)
predicted_class = model.predict(new_sample)
print("\nPrediction on new sample:")
print(f"Input: {new_sample[0]}")
print(f"Predicted class: {data.target_names[predicted_class][0]}")


# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(data.target_names))
plt.xticks(tick_marks, data.target_names, rotation=45)
plt.yticks(tick_marks, data.target_names)


# Adding text annotations
for i in range(len(data.target_names)):
    for j in range(len(data.target_names)):
        plt.text(j, i, conf_matrix[i, j], ha="center", va="center", color="red")


plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()





