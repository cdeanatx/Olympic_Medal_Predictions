# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import recall_score
import category_encoders as ce
from sklearn.compose import make_column_transformer

# Reading the dataset
olympic_data = pd.read_csv("./Resources/clean_regions_olympic_data.csv")

# printing the first few records
olympic_data.head()

# Columns in the dataset
olympic_data.columns.tolist()

# Getting data Stats
data_stats = olympic_data.describe()
data_stats

# Getting the null count columnwise
olympic_data.isnull().sum(axis=0)

# Delete region null records 
olympic_data = olympic_data[olympic_data['region'].notna()]
olympic_data.isnull().sum(axis=0)

olympic_data["medal"].value_counts()

olympic_data["medal"] = olympic_data["medal"].fillna(0)

olympic_data["is_medal"] = np.where(olympic_data["medal"] == 0, 0, 1)
olympic_data.head()

# Dropping Year, noc, city, sport, event, medal, region
olympic_data = olympic_data[["id","sex","season","event_category","region","is_medal","age","height","weight"]]
# add sport, noc/region

olympic_data.dtypes

olympic_data.shape

# ## Display Onehot and Label encoded results

# example of a one hot encoding
data = np.array(olympic_data[["sex","season","event_category"]])
print(data)
# define one hot encoding
encoder = OneHotEncoder(sparse=False)
# transform data
onehot = encoder.fit_transform(data)
print(onehot)
df = pd.DataFrame(onehot)
df.head()

# In df, columns 0&1 correspond to sex value, 2&3 correspond to season and other 52 correspond to event_category 

# example of a label encoding
olympic_data_encode = olympic_data.copy()
olympic_data_encode.head()

labelEncoder = preprocessing.LabelEncoder()
cat_cols = ["sex","season","event_category","region"]
mapping_dict ={}
for col in cat_cols:
    olympic_data_encode[col] = labelEncoder.fit_transform(olympic_data_encode[col])
 
    le_name_mapping = dict(zip(labelEncoder.classes_,
                        labelEncoder.transform(labelEncoder.classes_)))
 
    mapping_dict[col]= le_name_mapping
print(mapping_dict)

import csv
with open('test.csv', 'w') as f:
    for key in mapping_dict.keys():
        f.write("%s,%s\n"%(key,mapping_dict[key]))


# ## Split the data

# Split the data into train and test datasets for Winter
olympic_winter = olympic_data.loc[olympic_data["season"]=="Winter"]

X_winter = (olympic_winter.drop(["is_medal"], axis=1))
y_winter = np.array(olympic_winter["is_medal"])

Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_winter, y_winter, test_size=0.2, random_state=42)

# Split the data into train and test datasets for Summer
olympic_summer = olympic_data.loc[olympic_data["season"]=="Summer"]

X_summer = olympic_summer.drop(["is_medal"], axis=1)
y_summer = np.array(olympic_summer["is_medal"])

Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_summer, y_summer, test_size=0.2, random_state=42)

print(Xw_train.shape, Xw_test.shape, yw_train.shape, yw_test.shape)

print(Xs_train.shape, Xs_test.shape, ys_train.shape, ys_test.shape)


# ## Winter data with Randomforest & Onehot encoder

# Pre-processing Steps
ohe = OneHotEncoder()
model = RandomForestClassifier()
imputer = IterativeImputer()
scaler = StandardScaler()

column_tran = make_column_transformer(
                (OneHotEncoder(sparse=False,handle_unknown='ignore'),["sex","event_category","season", "region"]),
                remainder = "passthrough")

# Setup pipeline
pipe = make_pipeline(column_tran, imputer,scaler, model)

# Setup Cross-Validator
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# Fit the model with training dataset
pipe.fit(Xw_train, yw_train)

# Evaluate accuracy scores for the model
scores = cross_val_score(pipe, Xw_train, yw_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Predict the model with the test dataset
predictions = pipe.predict(Xw_test)
recall = recall_score(yw_test, predictions, average='binary')
print('Recall: %.3f' % recall)

# Calculating the confusion matrix.
cm = confusion_matrix(yw_test, predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

cm_df


# ## Winter data with Randomforest & Label encoder

# Pre-processing Steps
model = RandomForestClassifier()
imputer = IterativeImputer()
scaler = StandardScaler()

cat_cols = ["sex","season","event_category","region"]
Xw_train[cat_cols] = Xw_train[cat_cols].apply(preprocessing.LabelEncoder().fit_transform)
Xw_test[cat_cols] = Xw_test[cat_cols].apply(preprocessing.LabelEncoder().fit_transform)

# Setup pipeline
pipe = make_pipeline(imputer, scaler, model)

# Setup Cross-Validator
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# Fit the model with training dataset
pipe.fit(Xw_train, yw_train)

# Evaluate accuracy scores for the model
scores = cross_val_score(pipe, Xw_train, yw_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Predict the model with the test dataset
predictions = pipe.predict(Xw_test)
recall = recall_score(yw_test, predictions, average='binary')
print('Recall: %.3f' % recall)

# Calculating the confusion matrix.
cm = confusion_matrix(yw_test, predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

cm_df


# ## Summer data with Randomforest & Onehot encoder

# Pre-processing Steps
ohe = OneHotEncoder()
model = RandomForestClassifier()
imputer = IterativeImputer()
scaler = StandardScaler()

column_tran = make_column_transformer(
                (OneHotEncoder(sparse=False,handle_unknown='ignore'),["sex","event_category","season", "region"]),
                remainder = "passthrough")

# Setup pipeline
pipe = make_pipeline(column_tran, imputer,scaler, model)

# Setup Cross-Validator
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)


# Fit the model with training dataset
pipe.fit(Xs_train, ys_train)


# Evaluate accuracy scores for the model
#scores = cross_val_score(pipe, Xs_train, ys_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Predict the model with the test dataset
predictions = pipe.predict(Xs_test)
recall = recall_score(ys_test, predictions, average='binary')
print('Recall: %.3f' % recall)

# Calculating the confusion matrix.
cm = confusion_matrix(ys_test, predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

cm_df


# ## Summer data with Randomforest & Label encoder

# Pre-processing Steps
model = RandomForestClassifier()
imputer = IterativeImputer()
scaler = StandardScaler()

cat_cols = ["sex","season","event_category","region"]
Xs_train[cat_cols] = Xs_train[cat_cols].apply(preprocessing.LabelEncoder().fit_transform)
Xs_test[cat_cols] = Xs_test[cat_cols].apply(preprocessing.LabelEncoder().fit_transform)

# Setup pipeline
pipe = make_pipeline(imputer, scaler, model)

# Setup Cross-Validator
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# Fit the model with training dataset
pipe.fit(Xs_train, ys_train)

# Evaluate accuracy scores for the model
scores = cross_val_score(pipe, Xs_train, ys_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Predict the model with the test dataset
predictions = pipe.predict(Xs_test)
recall = recall_score(ys_test, predictions, average='binary')
print('Recall: %.3f' % recall)

# Calculating the confusion matrix.
cm = confusion_matrix(ys_test, predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

cm_df


# ## Check imputed values from IterativeImputer 

X = (olympic_data.drop(["is_medal"], axis=1))
y = np.array(olympic_data["is_medal"])

x_encoder = ce.TargetEncoder(cols=['sex', 'season', 'event_category','region'])
x_encoded = x_encoder.fit_transform(X,y)
imputer = IterativeImputer()
imputer.fit(x_encoded)
Xtrans = imputer.transform(x_encoded)

impute_df = pd.DataFrame(Xtrans, columns = ['id',"sex","season","event_category","region","age","height","weight"])
impute_df = impute_df.rename(columns={"age": "age_impute", "height": "height_impute","weight": "weight_impute"})
impute_df.head()

combined_df = pd.concat([x_encoded, impute_df[["age_impute","height_impute","weight_impute"]]], axis=1)

combined_df.to_csv("impute_data.csv", sep='\t')


# ## Label encode and then split the data to run the model
cat_cols = ["sex","season","event_category","region"]
olympic_data[cat_cols] = olympic_data[cat_cols].apply(preprocessing.LabelEncoder().fit_transform)

olympic_winter_enc = olympic_data.loc[olympic_data["season"]==1]

Xe_winter = (olympic_winter_enc.drop(["is_medal"], axis=1))
ye_winter = np.array(olympic_winter_enc["is_medal"])

Xew_train, Xew_test, yew_train, yew_test = train_test_split(Xe_winter, ye_winter, test_size=0.2, random_state=42)


olympic_summer_enc = olympic_data.loc[olympic_data["season"]==0]

Xe_summer = (olympic_summer_enc.drop(["is_medal"], axis=1))
ye_summer = np.array(olympic_summer_enc["is_medal"])

Xes_train, Xes_test, yes_train, yes_test = train_test_split(Xe_summer, ye_summer, test_size=0.2, random_state=42)



