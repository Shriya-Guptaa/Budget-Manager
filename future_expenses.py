import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

# Load the dataset
df = pd.read_csv('Daily Household Transactions.csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='mixed')

# Extract month and year features
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Create 'Category' column by extracting first word from 'Subcategory'
df['Category'] = df['Subcategory'].apply(lambda x: x.split()[0] if isinstance(x, str) else None)

# Group by 'Category' and calculate average expenses
category_avg_expenses = df.groupby('Category')['Amount'].mean().reset_index()

# Merge the average expenses with the original dataframe
df = pd.merge(df, category_avg_expenses, on='Category', suffixes=('', '_avg'))

# Calculate the difference between the amount and the average amount
df['Amount_diff'] = df['Amount'] - df['Amount_avg']

# Define bins based on the difference
bins = [-float('inf'), -150, 150, 300, float('inf')]
labels = ['Low', 'Medium', 'High', 'Very High']

# Create 'Expense_Level' based on the difference
df['Expense_Level'] = pd.cut(df['Amount_diff'], bins=bins, labels=labels)

# Convert 'Expense_Level' to categorical variable
df['Expense_Level'] = df['Expense_Level'].astype('category')

# Select relevant features for training
X = df[['Amount', 'Amount_avg']]
y = df['Expense_Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))

# Display a few predictions alongside the actual values
for i in range(10):
    print(f'Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}')

# Export the trained model using pickle
with open('expense_predictor_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("Model exported to expense_level_model.pkl")