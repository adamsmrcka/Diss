# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# Initialize the K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
predictions = knn.predict(X_test)

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
