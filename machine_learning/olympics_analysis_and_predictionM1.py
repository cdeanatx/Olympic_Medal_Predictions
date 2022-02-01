
# Importing the necessary libraries
import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime


from math import log

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Reading the dataset
olympic_data = pd.read_csv('./Resources/athlete_events.csv')

# printing the first few records
olympic_data.head()

# Columns available in the dataset
olympic_data_columns = olympic_data.columns.tolist()
olympic_data_columns

# Getting data Stats
data_stats = olympic_data.describe()
data_stats

# Getting the null count columnwise
column_null_counts = olympic_data.isnull().sum(axis=0)
column_null_counts

# Filling the NA Values in Medal to 'No Medal'
olympic_data['Medal'] = olympic_data['Medal'].fillna(value='No Medal')
olympic_data.head()

olympic_data = olympic_data.dropna()
olympic_data.shape


# Choosing the top five countries with the most data
countries = ['USA','BRA','GER','AUS','FRA']
sports = ['Athletics']

# Filtering on the most important features
olympic_data = olympic_data[olympic_data['NOC'].isin(countries)]
olympic_data = olympic_data[olympic_data['Season'] == 'Summer']

olympic_data = olympic_data[olympic_data['Height'].notna()]
olympic_data = olympic_data[olympic_data['Age'].notna()]
olympic_data['Height (m)'] = olympic_data['Height']/100
olympic_data = olympic_data[olympic_data['Weight'].notna()] 
olympic_data['BMI'] = round(olympic_data['Weight']/(olympic_data['Height (m)']*olympic_data['Height (m)']),2)
# olympic_data = olympic_data[olympics['Medal'] == 'Gold']
# wins = ['Gold','Bronze']
# olympic_data = olympic_data[olympic_data['Medal'].isin(wins)]

olympic_data.loc[(olympic_data['Medal'] == 'Gold'),'Medal']='Medal'
olympic_data.loc[(olympic_data['Medal'] == 'Silver'),'Medal']='Medal'
olympic_data.loc[(olympic_data['Medal'] == 'Bronze'),'Medal']='Medal'
olympic_data.loc[(olympic_data['Medal'].isna()),'Medal']= 'Non-Medal'

olympics_df = olympic_data[olympic_data['Year']< 2016]
olympics_2016 = olympic_data[olympic_data['Year'] == 2016]

# Using features with the highest importance
X = pd.get_dummies(olympics_df[["Sex", "Season"]])

y = olympics_df["Medal"]
print(X.shape, y.shape)

# test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# encoding on 2016 data
X_2016 = pd.get_dummies(olympics_2016[["Sex", "Season"]])
X_2016


y_2016 = olympics_2016["Medal"]
y_2016


classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# the closer the one the stronger the coorelation/prediction model
print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")


#Machine learning model predictions tested on 2016 data
# using only the 2016 Olympics dataset to see how close our predictions are
predictions = classifier.predict(X_2016)
print(f"First 10 Predictions:   {predictions[:10]}")
print(f"First 10 Actual labels: {y_2016[:10].tolist()}")


# putting actuals and predictions into a dataframe
Testing = pd.DataFrame({"Prediction": predictions, "Actual": y_2016}).reset_index(drop=True)
Testing


#Reviewing prediction data
Testing_crosstab = pd.crosstab(Testing['Actual'],Testing['Prediction'])
Testing_crosstab


Testing[(Testing['Prediction'] == 'Medal')]




