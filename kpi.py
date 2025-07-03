#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


# In[4]:


df = pd.read_csv(r"Downloads/financial_kpi_startup_dataset_700_rows.csv")  # Replace with your file name
selected_features = [
    'Marketing Spend ($)', 'New Customers', 'Lost Customers',
    'Cost of Goods Sold (COGS) ($)', 'Operating Expenses ($)',
    'Gross Margin (%)'
]
target = 'Revenue ($)'



# In[11]:


df.head()


# In[5]:


X = df[selected_features].copy()
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = XGBRegressor(random_state=42, verbosity=0)
model.fit(X_train, y_train)



# In[6]:


print("\n📊 Enter the following 6 Financial KPI values:")
marketing_spend = float(input("1. Marketing Spend ($): "))
new_customers = float(input("2. New Customers: "))
lost_customers = float(input("3. Lost Customers: "))
cogs = float(input("4. Cost of Goods Sold (COGS) ($): "))
operating_expenses = float(input("5. Operating Expenses ($): "))
gross_margin = float(input("6. Gross Margin (%): "))

sample_input = pd.DataFrame([{
    'Marketing Spend ($)': marketing_spend,
    'New Customers': new_customers,
    'Lost Customers': lost_customers,
    'Cost of Goods Sold (COGS) ($)': cogs,
    'Operating Expenses ($)': operating_expenses,
    'Gross Margin (%)': gross_margin
}])

sample_scaled = scaler.transform(sample_input)
predicted_revenue = model.predict(sample_scaled)[0]


# In[9]:


print(f"\n$ Predicted Revenue: ${predicted_revenue:,.2f}")
total_cost = marketing_spend + cogs + operating_expenses
net_profit = predicted_revenue - total_cost

if net_profit > 0:
    print("--->Result: This month ends in PROFIT 🎉")
elif net_profit < 0:
    print(">>> Result: This month ends in LOSS 💸")
else:
    print("$$ Result: This month is a BREAK-EVEN situation.")


# In[10]:


df['Predicted_Revenue_USD'] = model.predict(X_scaled)
df.to_csv("Financial_KPI_with_Predicted_Revenue.csv", index=False)
print("\n>>>CSV saved: Financial_KPI_with_Predicted_Revenue.csv")


# In[ ]:




