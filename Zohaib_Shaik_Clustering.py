# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load datasets
customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Merge datasets
merged_df = transactions_df.merge(customers_df, on='CustomerID', how='inner')
merged_df = merged_df.merge(products_df, on='ProductID', how='inner')

# Add recency and frequency analysis
from datetime import datetime
import pandas as pd

# Convert TransactionDate to datetime
merged_df['TransactionDate'] = pd.to_datetime(merged_df['TransactionDate'])

# Calculate recency and frequency
current_date = merged_df['TransactionDate'].max()
customer_metrics = merged_df.groupby('CustomerID').agg({
    'TransactionDate': lambda x: (current_date - x.max()).days,  # Recency
    'TransactionID': 'count',  # Frequency
    'TotalValue': 'sum'  # Monetary
}).rename(columns={
    'TransactionDate': 'Recency',
    'TransactionID': 'Frequency',
    'TotalValue': 'Monetary'
})

# Define churn threshold (90 days)
customer_metrics['Churned'] = customer_metrics['Recency'] > 90
# Define frequent buyers (more than 5 purchases)
customer_metrics['FrequentBuyer'] = customer_metrics['Frequency'] > 5

# Summary statistics
print("Customer Metrics Summary:")
print("\
Churn Rate:", (customer_metrics['Churned'].mean() * 100).round(2), "%")
print("Frequent Buyers:", (customer_metrics['FrequentBuyer'].mean() * 100).round(2), "%")

# Plot frequency distribution
plt.figure(figsize=(10, 6))
plt.hist(customer_metrics['Frequency'], bins=20)
plt.title('Distribution of Purchase Frequency')
plt.xlabel('Number of Purchases')
plt.ylabel('Number of Customers')
plt.show()

# Display top 5 most valuable customers
print("\
Top 5 Most Valuable Customers:")
print(customer_metrics.sort_values('Monetary', ascending=False).head())