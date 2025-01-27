# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Merge datasets
merged_df = transactions_df.merge(customers_df, on='CustomerID', how='inner')
merged_df = merged_df.merge(products_df, on='ProductID', how='inner')

# Convert TransactionDate to datetime
merged_df['TransactionDate'] = pd.to_datetime(merged_df['TransactionDate'])

# Calculate recency, frequency, and monetary value
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

# Calculate Average Order Value (AOV)
customer_metrics['AOV'] = customer_metrics['Monetary'] / customer_metrics['Frequency']

# Define segmentation rules
def assign_prime_category(row):
    if row['Frequency'] > 10 and row['AOV'] >= 500 and row['Recency'] <= 30:
        return 'Prime'
    elif row['Frequency'] > 5 and row['AOV'] >= 300 and row['Recency'] <= 90:
        return 'Sub-Prime'
    else:
        return 'Non-Prime'

customer_metrics['Category'] = customer_metrics.apply(assign_prime_category, axis=1)

# Summary statistics
category_summary = customer_metrics['Category'].value_counts()
print("Customer Category Distribution:")
print(category_summary)

# Visualization of category distribution
plt.figure(figsize=(8, 6))
category_summary.plot(kind='bar', color=['#4caf50', '#ff9800', '#f44336'], alpha=0.8, edgecolor='black')
plt.title('Customer Segmentation (Prime, Sub-Prime, Non-Prime)')
plt.xlabel('Category')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()

# Visualize metrics by category
metrics_by_category = customer_metrics.groupby('Category').agg({
    'Frequency': 'mean',
    'AOV': 'mean',
    'Recency': 'mean'
}).round(2)

# Plot metrics for comparison
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
colors = ['#4caf50', '#ff9800', '#f44336']

# Frequency comparison
metrics_by_category['Frequency'].plot(kind='bar', ax=ax[0], color=colors, edgecolor='black')
ax[0].set_title('Average Frequency by Category')
ax[0].set_ylabel('Frequency')

# AOV comparison
metrics_by_category['AOV'].plot(kind='bar', ax=ax[1], color=colors, edgecolor='black')
ax[1].set_title('Average Order Value (AOV) by Category')
ax[1].set_ylabel('AOV ($)')

# Recency comparison
metrics_by_category['Recency'].plot(kind='bar', ax=ax[2], color=colors, edgecolor='black')
ax[2].set_title('Average Recency by Category')
ax[2].set_ylabel('Recency (days)')

plt.tight_layout()
plt.show()

# Display segment-wise metrics
print("\nSegment Metrics:")
print(metrics_by_category)

# Example: Top 5 customers in each category
print("\nTop 5 Customers by Category:")
for category, group in customer_metrics.groupby('Category'):
    print(f"\nCategory: {category}")
    print(group.sort_values('Monetary', ascending=False).head())
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Merge datasets
merged_df = transactions_df.merge(customers_df, on='CustomerID', how='inner')
merged_df = merged_df.merge(products_df, on='ProductID', how='inner')

# Convert TransactionDate to datetime
merged_df['TransactionDate'] = pd.to_datetime(merged_df['TransactionDate'])

# Calculate recency, frequency, and monetary value
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

# Calculate Average Order Value (AOV)
customer_metrics['AOV'] = customer_metrics['Monetary'] / customer_metrics['Frequency']

# Define segmentation rules
def assign_prime_category(row):
    if row['Frequency'] > 10 and row['AOV'] >= 500 and row['Recency'] <= 30:
        return 'Prime'
    elif row['Frequency'] > 5 and row['AOV'] >= 300 and row['Recency'] <= 90:
        return 'Sub-Prime'
    else:
        return 'Non-Prime'

customer_metrics['Category'] = customer_metrics.apply(assign_prime_category, axis=1)

# Add customer names to metrics
customer_metrics = customer_metrics.merge(customers_df[['CustomerID', 'CustomerName']], on='CustomerID', how='left')

# Number of customers in each category
category_counts = customer_metrics['Category'].value_counts()
print("\nNumber of Customers in Each Category:")
print(category_counts)

# Scatter plot of AOV vs Frequency by category
plt.figure(figsize=(10, 6))
for category, data in customer_metrics.groupby('Category'):
    plt.scatter(data['Frequency'], data['AOV'], label=f"{category} Customers", alpha=0.7)

plt.title('Customer Segments (AOV vs Frequency)')
plt.xlabel('Frequency (Number of Purchases)')
plt.ylabel('AOV (Average Order Value)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Display sample customer data from each category
print("\nSample Customers from Each Category:")
for category, group in customer_metrics.groupby('Category'):
    print(f"\nCategory: {category}")
    print(group[['CustomerName', 'Frequency', 'AOV', 'Recency']].sort_values('Frequency', ascending=False).head())
