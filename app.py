import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Daily Household Transactions.csv')
categories = [
    'Transportation', 'Food', 'Subscription', 'Festivals', 'Other', 'Small Cap fund 2',
    'Small cap fund 1', 'Family', 'Equity Mutual Fund E', 'Apparel', 'Public Provident Fund',
    'Saving Bank account 1', 'Gift', 'Salary', 'Household', 'Dividend earned on Shares',
    'Interest', 'Life Insurance', 'Beauty', 'Health', 'Money Transfer', 'Maid', 'Culture',
    'Tax refund', 'Tourism', 'Share Market', 'Self-development', 'Amazon pay cashback',
    'Education', 'Scrap', 'Petty cash', 'Documents', 'Gpay Reward', 'Social Life',
    'Equity Mutual Fund A', 'Maturity amount', 'Fixed Deposit', 'Equity Mutual Fund C',
    'Equity Mutual Fund F', 'Recurring Deposit', 'Saving Bank account 2', 'Equity Mutual Fund D',
    'Equity Mutual Fund B', 'Bonus', 'Investment', 'Grooming', 'Rent', 'Cook', 'Garbage Disposal',
    'Water (Jar /Tanker)'
]
important_categories = [
    "Transportation", "Food", "Festivals", "Family", "Public Provident Fund", 
    "Saving Bank account 1", "Salary", "Household", "Health", "Interest", 
    "Life Insurance", "Rent", "Cook", "Water (jar /tanker)"
]

# Load model
expense_predictor_model = pickle.load(open('expense_predictor_model.pkl', 'rb'))

# Preprocess dataset
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Category'] = df['Subcategory'].apply(lambda x: x.split()[0] if isinstance(x, str) else None)
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Encode categorical features using the fitted LabelEncoders
    global le_mode,le_category,le_income_expense,le_currency,le_subcategory
    le_mode = LabelEncoder()
    le_category = LabelEncoder()
    le_income_expense = LabelEncoder()
    le_currency = LabelEncoder()
    le_subcategory = LabelEncoder()
    df['Mode'] = le_mode.fit_transform(df['Mode'])
    df['Category'] = le_category.fit_transform(df['Category'])
    df['Income/Expense'] = le_income_expense.fit_transform(df['Income/Expense'])
    df['Currency'] = le_currency.fit_transform(df['Currency'])
    le_subcategory.fit(df['Subcategory'])
    
    return df

df = preprocess_data(df)

# Fit the scaler
scaler = StandardScaler()
scaler.fit(df[['Amount', 'Month', 'Year', 'Mode', 'Income/Expense', 'DayOfWeek', 'Category', 'IsWeekend']])

# Custom CSS to improve the UI
st.markdown("""
    <style>
    .main { background-color: #9AAEEB; }
    .stButton>button { background-color: #608CEB; color: white; }
    .stTextInput>div>input { background-color: #212224; }
    .stNumberInput>div>input { background-color: #212224; }
    .stSelectbox>div>div>div { background-color: #212224; }
    .stDateInput>div>input { background-color: #212224; }
    </style>
""", unsafe_allow_html=True)

def categorize_transaction(transaction):
    if transaction['Income/Expense']=="Income":
        return 0
    # Preprocess transaction data
    transaction = pd.DataFrame(transaction, index=[0])
    # Load the original dataset
    df = pd.read_csv('Daily Household Transactions.csv')
    # Group by 'Category' and calculate average expenses
    category_avg_expenses = df.groupby('Category')['Amount'].mean().reset_index()
    # Get the average expense for the category
    category = transaction['Category'].iloc[0]
    avg_amount = category_avg_expenses.loc[category_avg_expenses['Category'] == category, 'Amount']
    if avg_amount.empty:
        avg_amount = 0  # or some other default value
    else:
        avg_amount = avg_amount.iloc[0]
    # Create a dataframe with the average amount
    avg_amount_df = pd.DataFrame({'Amount_avg': [avg_amount]})   
    # Select relevant features for expense level model
    X = transaction[['Amount']]
    
    # Concatenate the transaction dataframe with the average amount dataframe
    X = pd.concat([X, avg_amount_df], axis=1)
    
    # Use the trained model to predict expense level
    prediction = expense_predictor_model.predict(X)
    avg_amount=round(avg_amount,2)

    st.write(f"Average expense for {category}: {avg_amount}")
    st.write(f"Predicted expense level: {prediction[0]}")
    return avg_amount

def could_money_have_been_saved(transaction,avg_amount):
    if transaction["Income/Expense"] == "Income":
        st.write("Congratulations on the income!")
        return
    elif transaction['Category'] in important_categories:
        st.write("Necessities cannot be neglected for savings.")
        return
    elif avg_amount==0:
        return
    else:
        amount=transaction['Amount']
        if amount <= avg_amount + 100:
            st.write("Expense was within limits.")
        else:
            savings = amount - avg_amount+100
            st.write("You could have saved money by reducing the expense.")
            st.write(f"Possible savings: {savings}")

def main():
    st.title("Budget Manager")
    st.write("One place for all your finances!")
    
    # Input fields
    st.header("Add New Transaction")
    date = st.date_input("Date")
    mode = st.selectbox("Mode", ["Cash", "Card", "Online"])  
    category = st.selectbox("Category", categories)     
    subcategory = st.text_input("Subcategory")
    note = st.text_input("Note")
    amount = st.number_input("Amount", min_value=0.0, format="%.2f")
    income_expense = st.selectbox("Income/Expense", ["Income", "Expense"])
    currency = st.selectbox("Currency", ["INR"])
    
    # Create a dictionary to store transaction data
    transaction = {
        "Date": date,
        "Mode": mode,
        "Category": category,
        "Subcategory": subcategory,
        "Note": note,
        "Amount": amount,
        "Income/Expense": income_expense,
        "Currency": currency
    }
    
    # Add Transaction button
    if st.button("Add Transaction to record"):
        # Convert the transaction dictionary to a pandas dataframe
        transaction_df = pd.DataFrame([transaction])
        # Append the transaction to the daily household transactions CSV file
        transaction_df.to_csv('Daily Household Transactions.csv', mode='a', header=False, index=False)
        st.write("Record updated")  
     # Analyse button
    if st.button("Analyse"):
        could_money_have_been_saved(transaction,categorize_transaction(transaction))
    
        
if __name__ == "__main__":
    main()
