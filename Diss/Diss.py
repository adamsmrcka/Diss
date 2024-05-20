# Import necessary libraries
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = r'C:\Users\adams\OneDrive\Moje\data.csv'
df = pd.read_csv(file_path)

# Get the shape of the DataFrame
print("Data shape:", df.shape)

# Display the first few rows of the DataFrame
print("First few rows of data:")
print(df.head())

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Drop unnecessary column
df = df.drop(columns=['Unnamed: 32'])

# Apply label encoding to categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Print the label encoders
print("Label Encoders:")
print(label_encoders)

# Separate features and labels
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic regression functions
def predict(X, W, b):
    return np.dot(X, W) + b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, W, b):
    m = X.shape[0]
    z = predict(X, W, b)
    h = sigmoid(z)
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_step(X, y, W, b):
    m = X.shape[0]
    z = predict(X, W, b)
    h = sigmoid(z)
    error = h - y
    dj_dw = (1/m) * np.dot(X.T, error)
    dj_db = (1/m) * np.sum(error)
    return dj_dw, dj_db

def gradient_descent(X, y, W, b, alpha, iterations, record_interval):
    cost_history = [cost_function(X, y, W, b)]
    for i in range(iterations):
        dj_dw, dj_db = gradient_step(X, y, W, b)
        W -= alpha * dj_dw
        b -= alpha * dj_db
        if i % record_interval == 0:
            current_cost = cost_function(X, y, W, b)
            print(f"Iteration {i}: Cost = {current_cost}")
            cost_history.append(current_cost)
    return W, b, cost_history

# Initialize weights and hyperparameters
m, n = X_train.shape
W = np.zeros(n)
b = 0
alpha = 0.001
iterations = 5000
record_interval = 1000

# Perform gradient descent
W, b, cost_history = gradient_descent(X_train, y_train, W, b, alpha, iterations, record_interval)

# Print the final weights and bias
print('Final weights:', W)
print('Final bias:', b)

# Plot the cost change over iterations
plt.plot(range(0, iterations + 1, record_interval), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Change over Iterations')
plt.show()

# Make predictions on the test set
linear_combination = predict(X_test, W, b)
probabilities = sigmoid(linear_combination)
predictions = np.where(probabilities >= 0.5, 1, 0)

# Evaluate the model
cm = confusion_matrix(y_test, predictions)
accuracy = np.mean(predictions == y_test) * 100

# Print the confusion matrix and accuracy
print("Confusion Matrix:")
print(cm)
print(f'Accuracy: {accuracy:.2f}%')

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.text(0.5, 2.2, f'Accuracy: {accuracy:.2f}%', ha='center', va='center', transform=ax.transAxes, fontsize=12)
plt.show()
