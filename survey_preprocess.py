import pandas as  pd
import numpy as np
df = pd.read_csv('stack-overflow-developer-survey-2025/survey_results_public.csv', low_memory=False)
WaitForOneHotEncoder = [] # create a list to store the columns that need to be one-hot encoded later
WaitForExplode = {} # create a list to store the columns that need to be exploded later
WaitForFillNa = {} # create a dict  to store the columns that need to be filledna later
# don't overthink this function, it is just a fast way to explode a column with multiple values separated by a delimiter and one-hot encode the result, it is for machine learning data cleaning purpose, not for data analysis
def fast_explode(target_dataframe, target_column, fillna='', split=';', prefix='worked with', tmp_column_name='tmp_c'):
    # Step 1: Create a temporary column with split and prefix
    _exploded = (
        target_dataframe.assign(
            **{tmp_column_name: target_dataframe[target_column]
                .fillna(fillna)
                .str.split(split)
                .apply(lambda lst: [f"{prefix} {lang.strip()}" for lang in lst if lang])
            }
        )
        .explode(tmp_column_name)
    )

    # Step 2: One-hot encode
    _one_hot = pd.crosstab(index=_exploded.index, columns=_exploded[tmp_column_name]).astype(bool)

    # Step 3: Combine with original DataFrame
    print(target_column, _one_hot.shape[0], target_dataframe.shape[0])
    result = pd.concat([target_dataframe.drop(columns=[target_column]), _one_hot], axis=1)
    result[_one_hot.columns] = result[_one_hot.columns].fillna(False)

    return result

df.set_index('ResponseId', inplace=True) # setting index
WaitForOneHotEncoder.append("MainBranch")
WaitForOneHotEncoder.append("Age")
WaitForOneHotEncoder.append("RemoteWork")
df['RemoteWork'] = df['RemoteWork'].fillna('not filled')# filling null values with 0
WaitForOneHotEncoder.append("EdLevel")
df['EdLevel'] = df['EdLevel'].fillna('not filled')
# df.drop(columns=['Check'], inplace=True)
WaitForExplode["CodingActivities"]  = {'fillna':'', 'split':';', 'prefix':'coding_activities '}
WaitForOneHotEncoder.append("Employment")
WaitForFillNa['Employment'] = 'not filled'
for col in ['LearnCode', 'LearnCodeOnline', 'TechDoc']:
    WaitForExplode[col]  = {'fillna':'', 'split':';', 'prefix':f'{col} '}
df['YearsCode'] = df['YearsCode'].replace({'Less than 1 year': 0, 'More than 50 years': 51})
df['YearsCode'] = df['YearsCode'].fillna(-1)
df['YearsCode'] = df['YearsCode'].astype(int)
# df['YearsCodePro'] = df['YearsCodePro'].replace({'Less than 1 year': 0, 'More than 50 years': 51})
# df['YearsCodePro'] = df['YearsCodePro'].fillna(-1)
# df['YearsCodePro'] = df['YearsCodePro'].astype(int)
WaitForOneHotEncoder.append("DevType")
WaitForFillNa['DevType'] = 'not filled'
WaitForOneHotEncoder.append("OrgSize")
WaitForFillNa['OrgSize'] = 'not filled'
WaitForOneHotEncoder.append("PurchaseInfluence")
df['PurchaseInfluence'] = df['PurchaseInfluence'].fillna('not filled')
# df = fast_explode(df,target_column='BuyNewTool', fillna='', split=';', prefix='buy_new_tool ')
# WaitForOneHotEncoder.append("BuildvsBuy")
# # filling null values with -1, meaning?
# #TODO meaning?
# WaitForFillNa['BuildvsBuy'] = 'not filled'
drop_columns = []
explode_columns = []
for col in df.columns:
    if 'HaveWorkedWith' in col:
        explode_columns.append(col)

    elif  'Admired' in col or 'WantTo' in col or 'SO' in col or 'AI' in col:
        drop_columns.append(col)
for col in explode_columns:
    WaitForExplode[col]  = {'fillna':'', 'split':';', 'prefix':f'{col} '}
WaitForOneHotEncoder.append("ICorPM")
WaitForFillNa['ICorPM'] = 'not filled'
WaitForFillNa['WorkExp'] = -1
for i in range(1,10):
    WaitForOneHotEncoder.append(f'Knowledge_{i}')
    WaitForFillNa[f'Knowledge_{i}'] = 'not filled'

# Frequency_1 ~ 3 also
for i in range(1,4):
    WaitForOneHotEncoder.append(f'Frequency_{i}')
    WaitForFillNa[f'Frequency_{i}'] = 'not filled'
WaitForOneHotEncoder.append("TimeSearching")
WaitForFillNa['TimeSearching'] = 'not filled'
WaitForOneHotEncoder.append("TimeAnswering")
WaitForFillNa['TimeAnswering'] = 'not filled'
WaitForOneHotEncoder.append("Challenge_Frustration")
WaitForFillNa['Challenge_Frustration'] = 'not filled'
WaitForOneHotEncoder.append("Company_ProfessionalTech")
WaitForFillNa['Company_ProfessionalTech'] = 'not filled'
WaitForOneHotEncoder.append("ProfessionalCloud")
WaitForFillNa['ProfessionalCloud'] = 'not filled'
WaitForOneHotEncoder.append("FirstAnswerer_ProfessionalQuestion")
WaitForFillNa['FirstAnswerer_ProfessionalQuestion'] = 'not filled'
WaitForOneHotEncoder.append("Industry")
WaitForFillNa['Industry'] = 'not filled'
WaitForFillNa['CompTotal'] = 0
df['Currency'] = df['Currency'].str[:3]
top_20_currency_by_num_rate = {
  "EUR": 1.0822,     # 1 EUR = ~1.0822 USD :contentReference[oaicite:0]{index=0}
  "USD": 1.0,         # by definition
  "GBP": 1.2781,     # 1 GBP = ~1.2781 USD :contentReference[oaicite:1]{index=1}
  "INR": 0.01195,    # 1 INR = ~0.01195 USD :contentReference[oaicite:2]{index=2}
  "CAD": 1 / 1.3699, # from Fed G.5A: 1.3699 CAD = 1 USD → invert to ~0.7299 USD per CAD :contentReference[oaicite:3]{index=3}
  "BRL": 1 / 5.3872,  # 5.3872 BRL = 1 USD → ~0.1856 USD per BRL :contentReference[oaicite:4]{index=4}
  "PLN": 0.2512,      # 1 PLN = ~0.2512 USD :contentReference[oaicite:5]{index=5}
  "AUD": 0.6597,      # 1 AUD = ~0.6597 USD :contentReference[oaicite:6]{index=6}
  "SEK": 1 / 10.5744,  # 10.5744 SEK = 1 USD → ~0.0946 USD per SEK :contentReference[oaicite:7]{index=7}
  "CHF": 1 / 0.8808,   # 0.8808 CHF = 1 USD → ~1.1356 USD per CHF :contentReference[oaicite:8]{index=8}
  "CZK": 1 / 25.120,   # 25.120 CZK = 1 USD → ~0.03983 USD per CZK :contentReference[oaicite:9]{index=9}
  "DKK": 1 / 6.8955,    # 6.8955 DKK = 1 USD → ~0.1451 USD per DKK :contentReference[oaicite:10]{index=10}
  "NOK": 1 / 10.7574,   # 10.7574 NOK = 1 USD → ~0.09297 USD per NOK :contentReference[oaicite:11]{index=11}
  "ILS": 1 / 4.0067,    # 4.0067 ILS = 1 USD → ~0.2496 USD per ILS :contentReference[oaicite:12]{index=12}
  "NZD": 0.6050,        # 1 NZD = ~0.6050 USD :contentReference[oaicite:13]{index=13}
  "ZAR": 1 / 18.3346,   # 18.3346 ZAR = 1 USD → ~0.05456 USD per ZAR :contentReference[oaicite:14]{index=14}
  "MXN": 1 / 18.3062,   # 18.3062 MXN = 1 USD → ~0.05463 USD per MXN :contentReference[oaicite:15]{index=15}
  "TRY": 1 / 35.5734    # 35.5734 TRY = 1 USD → ~0.02812 USD per TRY :contentReference[oaicite:16]{index=16}
}
df = df[df['Currency'].isin(top_20_currency_by_num_rate.keys())]

# remove smaller than 1000 usd per year
df = df.loc[(df['CompTotal'] >= 1000)]
# remove outliners by total compensation keep quantile 1~99 data
lower_bound = df['CompTotal'].quantile(0.01)
upper_bound = df['CompTotal'].quantile(0.99)
df = df[(df['CompTotal'] >= lower_bound) & (df['CompTotal'] <= upper_bound)]

# remove outliners by currency keep quantile 1~99 data
for currency in df['Currency'].unique():
    currency_mask = df['Currency'] == currency
    lower_bound = df.loc[currency_mask, 'CompTotal'].quantile(0.01)
    upper_bound = df.loc[currency_mask, 'CompTotal'].quantile(0.99)
    df = df[~((currency_mask) & ((df['CompTotal'] < lower_bound) | (df['CompTotal'] > upper_bound)))]

# remove outliners by country keep quantile 1~99 data
for country in df['Country'].unique():
    country_mask = df['Country'] == country
    lower_bound = df.loc[country_mask, 'CompTotal'].quantile(0.01)
    upper_bound = df.loc[country_mask, 'CompTotal'].quantile(0.99)
    df = df[~((country_mask) & ((df['CompTotal'] < lower_bound) | (df['CompTotal'] > upper_bound)))]



df['CompTotal'] = df.apply(lambda row: row['CompTotal'] * top_20_currency_by_num_rate[row['Currency']] if row['Currency'] in top_20_currency_by_num_rate and row['CompTotal'] > 0 else row['CompTotal'], axis=1)
#remove salaries below 1000 usd per year ,remove larger than 300000 usd per year
df.drop(columns=drop_columns, inplace=True)
for col, params in WaitForFillNa.items():
    try:
        df[col] = df[col].fillna(params)
    except Exception as e:
        print(
            f"Error filling NaN for column {col} with value {params}: {e}"
        )
for col, params in WaitForExplode.items():
    try:
        df = fast_explode(df,target_column=col, fillna=params['fillna'], split=params['split'], prefix=params['prefix'])
    except Exception as e:
        print(
            f"Error filling NaN for column {col} with value {params}: {e}"
        )

df['seniority'] = pd.cut(df['WorkExp'].fillna(0), bins=[-0.1,1,3,7,15,50], labels=['Intern/Junior','Junior','Mid','Senior','Principal'])
WaitForOneHotEncoder.append('seniority')
# select top 20 countries which have the most data points
top_countries = df['Country'].value_counts().nlargest(20).index.tolist()
for country in df.Country.unique():
    log_comp = df[df.Country==country]['CompTotal'].apply(lambda x: np.log1p(x))
    Q1 = log_comp.quantile(0.25)
    Q3 = log_comp.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = log_comp[(log_comp < lower_bound) | (log_comp > upper_bound)]
    print(f'{country} start to remove outliers')
    print(f"Q1={Q1}, Q3={Q3}, IQR={IQR}")
    print(f"Lower bound={lower_bound}, Upper bound={upper_bound}")
    print(f"Outliers count={len(outliers)}")
    print('-'*40)
    # df = df.loc[(df['CompTotal'] >= 1000) & (df['CompTotal'] <= 500000)]
    # print(df[['CompTotal']].describe())
    # remove data using the outlier bounds' index
    df = df.loc[~df.index.isin(outliers.index)]
df.reset_index(inplace=True,drop=True)

for edlevel in df.EdLevel.unique():
    log_comp = df[df.EdLevel==edlevel]['CompTotal'].apply(lambda x: np.log1p(x))
    Q1 = log_comp.quantile(0.25)
    Q3 = log_comp.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = log_comp[(log_comp < lower_bound) | (log_comp > upper_bound)]
    print(f'{edlevel} start to remove outliers')
    print(f"Q1={Q1}, Q3={Q3}, IQR={IQR}")
    print(f"Lower bound={lower_bound}, Upper bound={upper_bound}")
    print(f"Outliers count={len(outliers)}")
    print('-'*40)
    # df = df.loc[(df['CompTotal'] >= 1000) & (df['CompTotal'] <= 500000)]
    # print(df[['CompTotal']].describe())
    # remove data using the outlier bounds' index
    df = df.loc[~df.index.isin(outliers.index)]
df.reset_index(inplace=True,drop=True)
for i in range(1, 12):
    if i in [2,3]:
        continue
    df[f'JobSatPoints_{i}'] = df[f'JobSatPoints_{i}'].fillna(-1)
    WaitForFillNa[f'JobSatPoints_{i}'] = -1

total_score = df[[f'JobSatPoints_{i}' for i in range(1, 12) if i not in [2,3]]].sum(axis=1)
df.loc[total_score > 100, [f'JobSatPoints_{i}' for i in range(1, 12) if i not in [2,3]]] = df.loc[total_score > 100, [f'JobSatPoints_{i}' for i in range(1, 12) if i not in [2,3]]].div(total_score[total_score > 100], axis=0).multiply(100)
df[[f'JobSatPoints_{i}' for i in range(1, 12) if i not in [2,3]]] = df[[f'JobSatPoints_{i}' for i in range(1, 12) if i not in [2,3]]].round().astype(int)

total_score = df[[f'JobSatPoints_{i}' for i in range(1, 12) if i not in [2,3]]].sum(axis=1)
print(total_score.value_counts())
# ok there are some value larger than 100, let's check total number
print((total_score > 100).sum())
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0])
drop_columns.extend(['TechEndorse','SurveyLength','SurveyEase','ConvertedCompYearly'])
explode_columns = ['OpSysPersonal use','OpSysProfessional use']
for col in explode_columns:
    df = fast_explode(df,target_column=col, fillna='', split=';', prefix=f'{col} ')

for col in drop_columns:
    try:
        df.drop(columns=[col], inplace=True)
    except Exception as e:
        print(f"Error dropping column {col}: {e}")
# df['TBranch'] = df['TBranch'].map({'Yes': 1, 'No': 0})
# df['TBranch'] = df['TBranch'].fillna(-1)

WaitForOneHotEncoder.append('Country')
WaitForOneHotEncoder.append('Currency')
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0])
df.replace({True: 1, False: 0}, inplace=True)
df_cleaned = df.copy()
df_cleaned.dropna(inplace=True)
df = df[[c for c in df.columns if "JobSat" not in c]].copy()
usa = df[(df["Country"] == "United States of America") & (df["Currency"] == "USD")].copy()
# If you don't want to filter currency, use: usa = df[df["Country"] == "United States of America"].copy()
usa = usa[usa["CompTotal"] >= 1000]
usa.to_csv('2025_survey.csv',index=False)