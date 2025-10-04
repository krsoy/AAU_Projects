import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
coffee_df = pd.read_csv('coffee_df.csv')
origin_df = coffee_df.copy()
coffee_df.to_csv('coffee_df.csv')
'''
['submission_id', 'age', 'cups', 'where_drink', 'brew', 'brew_other',
       'purchase', 'purchase_other', 'favorite', 'favorite_specify',
       'additions', 'additions_other', 'dairy', 'sweetener', 'style',
       'strength', 'roast_level', 'caffeine', 'expertise',
       'coffee_a_bitterness', 'coffee_a_acidity',
       'coffee_a_personal_preference', 'coffee_a_notes', 'coffee_b_bitterness',
       'coffee_b_acidity', 'coffee_b_personal_preference', 'coffee_b_notes',
       'coffee_c_bitterness', 'coffee_c_acidity',
       'coffee_c_personal_preference', 'coffee_c_notes', 'coffee_d_bitterness',
       'coffee_d_acidity', 'coffee_d_personal_preference', 'coffee_d_notes',
       'prefer_abc', 'prefer_ad', 'prefer_overall', 'wfh', 'total_spend',
       'why_drink', 'why_drink_other', 'taste', 'know_source', 'most_paid',
       'most_willing', 'value_cafe', 'spent_equipment', 'value_equipment',
       'gender', 'gender_specify', 'education_level', 'ethnicity_race',
       'ethnicity_race_specify', 'employment_status', 'number_children',
       'political_affiliation']
'''
print(coffee_df.head())
print(coffee_df.columns)
print(coffee_df.dropna().info())
# count all column null values
print(coffee_df.isnull().sum())

# select column need to drop for too many null values
need_to_drop = coffee_df.isnull().sum()[coffee_df.isnull().sum() > 1000].index.tolist()
# drop the columns
coffee_df.drop(columns=need_to_drop,inplace=True)

# output check value counts for each column, check what type of data in each column
for c in coffee_df.columns:
    # output value counts for each column
    print(f'Column: {c}')
    print(coffee_df[c].value_counts())
    print('*'*40)

# now we found that some column have multiple value in one cell, need to separate them into multiple columns
column_need_to_separate = ['where_drink', 'why_drink','brew','additions']

# create a list to record all new column names after separate, so that we can keep them later
column_need_to_keep = []
# start separate
for c in column_need_to_separate:
    separate_df  = coffee_df[c].str.get_dummies(sep=', ')
    separate_df.columns = [f'{c}_{col}' for col in separate_df.columns]
    print(separate_df.head())
    # extend not append
    column_need_to_keep.extend(separate_df.columns)
    coffee_df = pd.concat([coffee_df, separate_df], axis=1)

print(coffee_df.head())

# we need to drop original column_need_to_separate column
coffee_df.drop(columns=column_need_to_separate,inplace=True)
print(coffee_df.info())

# transform all object type column to category type column
# needs to keep value maps for further representation
value_maps = {}
# all columns except column_need_to_separate complete column, need to replace their value with unique integer match unique value
for c in coffee_df.columns:
    if c not in column_need_to_separate:
        if c not in column_need_to_keep:
            unique_values = coffee_df[c].unique()
            value_map = {v: i for i, v in enumerate(unique_values)}
            value_maps[c] = value_map
            coffee_df[c] = coffee_df[c].map(value_map)


print(coffee_df.head())


plt.figure(figsize=(20, 15))
# calculate correlation matrix
corr = coffee_df.corr()
# plot heatmap
# show numbers on each cell
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

plt.colorbar() # add color bar
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns )
plt.title('Correlation Heatmap')
plt.tight_layout()
# show
plt.show()

# coffe amount by age

# replace column age value using value_maps
for c in ['age', 'expertise', 'know_source','total_spend']:
    if c in value_maps:
        tmp_map = value_maps[c]
        reversed_map = {v: k for k, v in tmp_map.items()}
        coffee_df[c] = coffee_df[c].map(reversed_map)

tmp_df = coffee_df[['know_source', 'expertise','total_spend']].dropna()
tmp_df.groupby('know_source')['expertise'].mean().plot(kind='bar')


def convert_spend_range(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if "-" in x:
        low, high = x.replace("$", "").split("-")
        return (float(low) + float(high)) / 2
    elif ">" in x:
        return float(x.replace("$", "").replace(">", "")) + 10
    elif "<" in x:
        return float(x.replace("$", "").replace("<", "")) / 2
    else:
        return float(x.replace("$", ""))

coffee_df['total_spend'] = tmp_df['total_spend'].apply(convert_spend_range)

# fig,axs = plt.subplots(2,1,figsize=(10,10))
# age_count = coffee_df['age'].value_counts().sort_index()
# axs[0].bar(age_count.index,age_count.values)
# axs[0].set_title('Age Distribution')
# axs[0].set_xlabel('Age')
# axs[0].set_ylabel('Counts')
#
# age_cup_mean = coffee_df.groupby('age')['cups'].mean()
# axs[1].bar(age_cup_mean.index,age_cup_mean.values,)
# axs[1].set_title('Average Cups by Age')
# axs[1].set_xlabel('Age')
# axs[1].set_ylabel('Average Cups')
# plt.show()


dc = coffee_df[['expertise', 'know_source','total_spend']].dropna()


from scipy import stats
# know source need to be replaced with boolean/int value
unique_values = dc['know_source'].unique()
value_map = {v: i for i, v in enumerate(unique_values)}
value_maps[c] = value_map
dc['know_source'] = dc['know_source'].map(value_map)
# loop comparison
done = []
