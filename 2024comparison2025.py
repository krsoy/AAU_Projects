import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_24 = pd.read_csv('stack-overflow-developer-survey-2024/survey_results_public.csv')
data_25 = pd.read_csv('stack-overflow-developer-survey-2025/survey_results_public.csv')
usa_24 = pd.read_csv('usa_salary_data.csv')
print(data_24.columns)
print(data_25.columns)
common_cols = set(data_24.columns).intersection(set(data_25.columns))
usa_common_cols = set(usa_24.columns).intersection(set(data_25.columns))
print(common_cols)
print(usa_common_cols)