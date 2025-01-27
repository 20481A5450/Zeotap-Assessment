# --------------------------------------
# 1. Import Libraries and Load Datasets
# --------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
customers_df = pd.read_csv("Customers.csv")
products_df = pd.read_csv("Products.csv")
transactions_df = pd.read_csv("Transactions.csv", encoding='utf-8', on_bad_lines='warn')

# Display dataset overviews
print("\nCustomers Dataset Overview:")
print(customers_df.head())
print("\nProducts Dataset Overview:")
print(products_df.head())
print("\nTransactions Dataset Overview:")
print(transactions_df.head())

# --------------------------------------
# 2. Data Preprocessing
# --------------------------------------

# Convert date columns to datetime
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])

# --------------------------------------
# 3. Exploratory Data Analysis (EDA)
# --------------------------------------

# Sales Trend Analysis
monthly_sales = transactions_df.groupby(transactions_df['TransactionDate'].dt.to_period('M'))['TotalValue'].sum().reset_index()
monthly_sales['TransactionDate'] = monthly_sales['TransactionDate'].astype(str)

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['TransactionDate'], monthly_sales['TotalValue'])
plt.xticks(rotation=45)
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales Value')
plt.tight_layout()
plt.show()

# Top Categories Analysis
top_categories = pd.merge(transactions_df, products_df, on='ProductID')
category_sales = top_categories.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
category_sales.plot(kind='bar')
plt.title('Sales by Product Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Regional Analysis
customer_region_sales = pd.merge(transactions_df, customers_df, on='CustomerID')
region_sales = customer_region_sales.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
region_sales.plot(kind='bar')
plt.title('Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Customer Metrics Analysis
customer_metrics = transactions_df.groupby('CustomerID').agg({
    'TransactionID': 'count',
    'TotalValue': 'sum',
    'Quantity': 'sum'
}).rename(columns={
    'TransactionID': 'total_transactions',
    'TotalValue': 'total_spend',
    'Quantity': 'total_items'
})
customer_metrics['avg_order_value'] = customer_metrics['total_spend'] / customer_metrics['total_transactions']

# Regional Performance Metrics
regional_metrics = customer_region_sales.groupby('Region').agg({
    'TotalValue': 'sum',
    'TransactionID': 'nunique',
    'CustomerID': 'nunique'
}).round(2)
regional_metrics['avg_customer_value'] = (regional_metrics['TotalValue'] / 
                                          regional_metrics['CustomerID']).round(2)

# Customer Recency Analysis
latest_date = transactions_df['TransactionDate'].max()
customer_recency = transactions_df.groupby('CustomerID')['TransactionDate'].max()
customer_recency = (latest_date - customer_recency).dt.days

# --------------------------------------
# 4. Business Insights
# --------------------------------------

print("\nKey Business Insights:")
insights = [
    "1. Sales Trend: Monthly sales show a steady increase, with an average monthly growth rate of {:.2f}%.".format(
        monthly_sales['TotalValue'].pct_change().mean() * 100),
    "2. Top Selling Category: '{}' generates the highest revenue with ${:.2f} in sales.".format(
        category_sales.index[0], category_sales.iloc[0]),
    "3. Most Valuable Region: '{}' leads in revenue with ${:.2f} in sales.".format(
        region_sales.index[0], region_sales.iloc[0]),
    "4. Customer Spending: Average transaction value is ${:.2f}.".format(
        transactions_df['TotalValue'].mean()),
    "5. Customer Recency: The average time since last transaction is {:.2f} days.".format(
        customer_recency.mean()),
    "6. Regional Insights: Highest average customer value is ${:.2f} in the '{}' region.".format(
        regional_metrics['avg_customer_value'].max(), regional_metrics['avg_customer_value'].idxmax())
]

for insight in insights:
    print(insight)

# --------------------------------------
# 5. Predictive Modeling
# --------------------------------------

# Preparing data for modeling
model_data = transactions_df[['Quantity', 'Price', 'TotalValue']]
X = model_data[['Quantity', 'Price']]
y = model_data['TotalValue']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("\nPredictive Model Performance:")
print("Mean Squared Error: {:.2f}".format(mean_squared_error(y_test, y_pred)))
print("R-squared: {:.2f}".format(r2_score(y_test, y_pred)))

# --------------------------------------
# 6. Advanced Analysis
# --------------------------------------

# Top 10 Customers Analysis
customer_spending = transactions_df.groupby('CustomerID')['TotalValue'].sum().sort_values(ascending=False)
total_revenue = customer_spending.sum()
top_10_customers = customer_spending.head(10)
top_10_percentage = (top_10_customers.sum() / total_revenue) * 100

print("Top 10 Customers Analysis:")
print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Top 10 Customers Revenue: ${top_10_customers.sum():,.2f}")
print(f"Top 10 Customers Percentage: {top_10_percentage:.2f}%")

# Visualize top customer contribution
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), top_10_customers.values)
plt.title('Top 10 Customers Revenue Contribution')
plt.xlabel('Customer Rank')
plt.ylabel('Revenue ($)')
plt.xticks(range(1, 11))
plt.tight_layout()
plt.show()

# Regional product performance
regional_performance = pd.merge(
    transactions_df, 
    customers_df[['CustomerID', 'Region']], 
    on='CustomerID'
).merge(
    products_df[['ProductID', 'ProductName']], 
    on='ProductID'
)

region_product_revenue = regional_performance.groupby(['Region', 'ProductName'])['TotalValue'].sum().reset_index()

print("Top Product by Region:")
print(region_product_revenue.sort_values('TotalValue', ascending=False).groupby('Region').head(1))
