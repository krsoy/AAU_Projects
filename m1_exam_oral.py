import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
def bin_distribution(column, bins=10,rotate=0):
    plt.figure(figsize=(12,6))
    sns.histplot(df[column], bins=bins, kde=False)
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Count')
    # rotate x labels
    if rotate:
        plt.xticks(rotation=rotate)
    plt.show()

df = pd.read_csv('roman_data.csv')
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
df.dropna(inplace=True)
# all columns bin plot in one figure
# for col in df.select_dtypes(include=['int64', 'float64']).columns:
#     print(col)
#     bins = len(df[col].unique())+1
#     bin_distribution(col, bins=bins, rotate=45)

# predict type_dev column using other columns, is a classification problem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
# encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])
print(df.info())

results = {}
for col in df.select_dtypes(include=['int64', 'float64']).columns:

    X = df.drop(col, axis=1)
    y = df[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # baseline model
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)
    print(confusion_matrix(y_test, y_dummy))
    print(classification_report(y_test, y_dummy))
    # compare two models
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print("Dummy Classifier Accuracy:", accuracy_score(y_test, y_dummy))
    results[col] = {
        'target':col,
        'Random Forest Accuracy': accuracy_score(y_test, y_pred),
        'Dummy Classifier Accuracy': accuracy_score(y_test, y_dummy)
    }

results_df = pd.DataFrame(results).T
print(results_df)