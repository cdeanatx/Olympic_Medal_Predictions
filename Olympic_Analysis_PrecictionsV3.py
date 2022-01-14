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

olympic_data["medal"] = olympic_data["medal"].fillna(0)

olympic_data["is_medal"] = np.where(olympic_data["medal"] == 0, 0, 1)
olympic_data

olympic_data.isnull().sum(axis=0)

olympic_data = olympic_data[["id","sex","year","season","sport","is_medal","age","height","weight"]]

olympic_data.dtypes

# Split the data
X = olympic_data.drop(["is_medal"], axis=1)
y = np.array(olympic_data["is_medal"])

X = X.drop(["year"], axis=1)


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
from sklearn import preprocessing
import category_encoders as ce


# ### Using One Hot Encoder

# Taking very long time to run

# # define modeling pipeline
# model = RandomForestClassifier()
# imputer = IterativeImputer()
# pipeline = Pipeline(steps=[("onehot", OneHotEncoder(sparse=False,handle_unknown='ignore')),
#                            ('i', imputer), 
#                            ('m', model)
#                           ])

# # define model evaluation
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# # evaluate model
# scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# ### Using Target Encoder

# define modeling pipeline
model = RandomForestClassifier()
imputer = IterativeImputer()
pipeline = Pipeline(steps=[("targetEncoder", ce.TargetEncoder(cols=['sex', 'season', 'sport'])),
                           ('i', imputer), 
                           ('m', model)
                          ])

# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# ## Add Predict method and validate the data

# fit the pipeline with the training data
pipeline.fit(X,y)

# predict target values on the training data
x_predict = pipeline.predict(X)

df = pd.DataFrame(x_predict, columns = ['medal_predictions'])
df.value_counts()

# Actual medal column
olympic_data["is_medal"].value_counts()


# ## Check imputed values from IterativeImputer 

x_encoder = ce.TargetEncoder(cols=['sex', 'season', 'sport'])

x_encoded = x_encoder.fit_transform(X,y)
x_encoded

x_encoded.isnull().sum(axis=0)

imputer = IterativeImputer()

imputer.fit(x_encoded)

Xtrans = imputer.transform(x_encoded)

impute_df = pd.DataFrame(Xtrans, columns = ['id',"sex","season","sport","age","height","weight"])
impute_df

impute_df.isnull().sum(axis=0)

combined_df = pd.concat([x_encoded, impute_df[["age","height","weight"]]], axis=1)
combined_df[combined_df.isna().any(axis=1)]


# ## Use make_pipeline method 

# ## Add recall

# ## Add Ordinal regression to update the type of medal

# ## Add event category and remove sport

# ## Make sure the model runs for all records



