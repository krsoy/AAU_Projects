import pandas as pd
df = pd.read_('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-09-03/stackoverflow_survey_single_response.csv')
print(df.head())
print(df.describe())
print(df.info())
print(3)