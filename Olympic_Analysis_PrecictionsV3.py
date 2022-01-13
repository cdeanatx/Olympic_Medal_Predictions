# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Reading the dataset
olympic_data = pd.read_csv("./Resources/recent_olympic_data.csv")

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

olympic_data["medal"].value_counts()
# 172495 null values

# Replace NAs in medal column with 0
olympic_data["medal"] = olympic_data["medal"].fillna(0)

# Replace medal with 1
olympic_data["is_medal"] = np.where(olympic_data["medal"] == 0, 0, 1)
olympic_data

# Check for null values in all columns
olympic_data.isnull().sum(axis=0)

# Remove event column from the dataframe
olympic_data = olympic_data[["id","sex","year","season","sport","is_medal","age","height","weight"]]

# Check datatypes for each column
olympic_data.dtypes

# Split the data
X = olympic_data.drop(["is_medal"], axis=1).loc[0:999]
y = np.array(olympic_data["is_medal"])[:1000]

# imports for encoding, imputing and modeleing 
from numpy import isnan
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# define modeling pipeline
model = RandomForestClassifier()
imputer = IterativeImputer()
pipeline = Pipeline(steps=[("onehot", OneHotEncoder(sparse=False,handle_unknown='ignore')),
                           ('i', imputer), 
                           ('m', model)
                          ])

# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# Things to work on

# 1. Use make_pipeline method 

# 2. Check imputed values from IterativeImputer 

# 3. Add Predict method and validate the data

# 4. Add Ordinal regression to update the type of medal

# 5. Add event category and remove sport

# 6. Make sure the model runs for all records

