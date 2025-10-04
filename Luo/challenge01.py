import pandas as pd

# 1. Data for the challenge
sales_data = {
    'ProductID': ['A101', 'B202', 'C303', 'A101', 'D404'],
    'Price': [10.0, 25.5, 5.75, 9.5, 15.0],
    'UnitsSold': [15, 8, 20, 12, 6]
}

# Create the DataFrame
sales_df = pd.DataFrame(sales_data)
print("--- Initial DataFrame ---")
print(sales_df)

# 2. Calculate the 'Revenue' column
sales_df['Revenue'] = sales_df['Price'] * sales_df['UnitsSold']
print(sales_df)

# 3. Find all sales with more than 10 units sold
print(sales_df[sales_df['UnitsSold'] > 10])


# 4. Calculate the total revenue
print(sales_df['Revenue'].sum())