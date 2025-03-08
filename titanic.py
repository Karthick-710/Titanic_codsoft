# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the Titanic dataset
file_path = "Titanic.csv"  # Make sure this file is in the same directory as your script

# Load the dataset
df = pd.read_csv(file_path)

# Step 2: Data Exploration
print("First few rows of the dataset:")
print(df.head())
print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nBasic statistics of the dataset:")
print(df.describe())

# Visualize the distribution of survivors
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# Step 3: Data Preprocessing
# Drop columns that won't be used
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)

# Convert categorical variables to numerical
df = pd.get_dummies(df, columns=['Sex', 'Pclass'], drop_first=True)

# Define features and target variable
X = df.drop('Survived', axis=1)
y = df['Survived']

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 8: User Input for Prediction
print("\nEnter passenger details for prediction:")
sex = input("Gender (male or female): ").strip().lower()
age = float(input("Age: "))
pclass = int(input("Ticket class (1, 2, or 3): "))

# Convert sex to numerical
sex_num = 1 if sex == 'female' else 0

# Create a DataFrame for the input
input_data = pd.DataFrame({
    'Sex_male': [sex_num],
    'Pclass_2': [1 if pclass == 2 else 0],
    'Pclass_3': [1 if pclass == 3 else 0],
    'Age': [age],
})

# Ensure the input data has the same columns as the training data
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Make prediction
survival_prediction = model.predict(input_data)

# Output the prediction
if survival_prediction[0] == 1:
    print(f"The passenger is predicted to have survived.")
else:
    print(f"The passenger is predicted not to have survived.")
