
# Importing the necessary libraries
import warnings
warnings.simplefilter('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from numpy import isnan
import numpy as np


# Reading the dataset
olympic_data = pd.read_csv("./Resources/cleaned_olympic_data.csv", na_values='NA')


# printing the first few records
olympic_data.head()


# Columns available in the dataset
olympic_data_columns = olympic_data.columns.tolist()
olympic_data_columns


# Data type of columns
olympic_data.dtypes


# Getting data Stats
data_stats = olympic_data.describe()


# Getting the null count columnwise
col_null_counts = olympic_data.isnull().sum(axis=0)
col_null_counts


# Filling the NA Values in Medal to 'No Medal'
olympic_data['medal'] = olympic_data['medal'].fillna(value='No Medal')
olympic_data.head()


# The column Year doesn't have missing values but 
# we will include it since it might be helpful modeling the other three columns. 
# The age, height and weight could change across years


# Create an IterativeImputer object and set its min_value and max_value 
# params to be the min and max of corresponding columns


# list of columns to be imputed
cols_to_impute = ['year','age','height','weight']
# create IterativeImputer object and set min and max value parameters
iter_imp = IterativeImputer(min_value=olympic_data[cols_to_impute].min(), max_value=olympic_data[cols_to_impute].max())
# apply the imputer to fit and transform
imputed_cols = iter_imp.fit_transform(olympic_data[cols_to_impute])
# Assign the imputed array back to the original dataframe
olympic_data[cols_to_impute] = imputed_cols


# split into input and output elements
data = olympic_data.values
ix = [i for i in range(data.shape[1]) if i != 12]
X, y = data[:, ix], data[:, 12]
# print total missing
print('Missing: %d' % sum(pd.isnull(X).flatten()))

# 271 region missing values

