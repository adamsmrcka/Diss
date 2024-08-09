# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

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

# Feature scaling (optional for Random Forest, but can help with convergence in some cases)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

# Print accuracy
print(f'Random Forest Accuracy: {accuracy*100:.2f}%')

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Random Forest Confusion Matrix\nAccuracy: {accuracy*100:.2f}%')
plt.show()
