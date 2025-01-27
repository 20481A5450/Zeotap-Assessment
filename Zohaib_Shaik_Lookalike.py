import pandas as pd
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from decimal import Decimal, ROUND_DOWN

# Load data
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

merged = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")
customer_features = merged[["CustomerID", "Price_x", "Quantity", "TotalValue"]]

scl = StandardScaler()
# Encode and scale features if required
print(merged.head())
features = merged[["Price_x", "Quantity", "TotalValue"]]
scaled_features = scl.fit_transform(features)

knn = NearestNeighbors(n_neighbors=4, metric="cosine")
knn.fit(scaled_features)
distances, indices = knn.kneighbors(scaled_features)

lookalike_map = {}
for idx, customer_id in enumerate(customer_features["CustomerID"][:20]):
    similar_customers = []
    for i in range(1, 4):
        score_ = Decimal(1 - distances[idx][i]) * Decimal(100)
        score = score_.quantize(Decimal('0.01'), rounding=ROUND_DOWN)
        similar_customers.append({
            "cust_id": customer_features["CustomerID"].iloc[indices[idx][i]],
            "score": float(score)
        })
    lookalike_map[customer_id] = similar_customers

output = pd.DataFrame({"cust_id": lookalike_map.keys(), "lookalikes": [json.dumps(v) for v in lookalike_map.values()]})
output.to_csv("Zohaib_Shaik_Lookalike.csv", index=False)
