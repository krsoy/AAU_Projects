import seaborn as sns
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

data = sns.load_dataset("tips")
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())
print(data.duplicated().sum())
# find duplicated rows
print(data[data.duplicated()])
# it's all good, 1 duplicated row, considering the data size, we can keep it

# create a tip_percent Variable
tip_percent = data['tip'] / data['total_bill']
data['tip_percent'] = tip_percent
print(data['tip_percent'].describe())


# plot average tip by day
avg_tip_by_day = data.groupby('day')['tip'].mean().reset_index()
print(avg_tip_by_day)
plt.figure(figsize=(8,5))
sns.barplot(data=avg_tip_by_day, x='day', y='tip')
plt.title('Average Tip by Day')
plt.xlabel('Day')
plt.ylabel('Average Tip')
plt.show()

# conclusion: average tip is highest on Saturday, lowest on Friday. But the difference is not significant.


# plot total bill vs tip
plt.figure(figsize=(8,5))
sns.regplot(data=data, x='total_bill', y='tip')
plt.title('Total Bill vs Tip by Day')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

# corr and p-value
corr = data['total_bill'].corr(data['tip'])
print("Correlation between total_bill and tip:", corr)
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(data['total_bill'], data['tip'])
print("P-value:", p_value)
# conclusion: there is a positive correlation between total bill and tip, as total bill increases, tip also increases.


# plot distribution of total bill
plt.figure(figsize=(8,5))
sns.histplot(data=data, x='total_bill', bins=20, kde=True)
plt.title('Distribution of Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.show()

# conclusion: total bill is right skewed, most of the total bill is between 10 and 30.
# output the skewness and kurtosis
print("Skewness of total_bill:", data['total_bill'].skew())


# plot tip by smoker
plt.figure(figsize=(8,5))
sns.boxplot(data=data, x='smoker', y='tip')
plt.title('Tip by Smoker')
plt.xlabel('Smoker')
plt.ylabel('Tip')
plt.show()
# conclusion: smokers tend to tip a little bit more than non-smokers, but the difference is not significant.





