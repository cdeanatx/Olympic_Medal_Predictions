# Importing the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.metrics import recall_score
from imblearn.ensemble import EasyEnsembleClassifier 

# Reading the dataset
olympic_data = pd.read_csv("./Resources/final_olympic_data.csv")

# printing the first few records
print(olympic_data.head())
# Getting data Stats
print(olympic_data.describe())

# Getting the null count columnwise
print(olympic_data.isnull().sum(axis=0))

# Replace nulls in medal column with 0
olympic_data["medal"] = olympic_data["medal"].fillna(0)

# Replace medals with 1
olympic_data["is_medal"] = np.where(olympic_data["medal"] == 0, 0, 1)
print(olympic_data.head())

# Dropping id, year, noc, city, sport, event, medal
olympic_data = olympic_data[["sex","season","event_category","region","is_medal","age","height","weight"]]

# Check the data types of olympic_data dataframe
print(olympic_data.dtypes)

# Check the shape of olympic_data dataframe
print(olympic_data.shape)


# ## Easy Ensemble Classifier
# ### Encode the categogical columns using Label encoding
olympic_data_encode = olympic_data.copy()

cat_cols = ["sex","season","event_category","region"]
olympic_data[cat_cols] = olympic_data[cat_cols].apply(preprocessing.LabelEncoder().fit_transform)

# Display label encoded values
labelEncoder = preprocessing.LabelEncoder()

mapping_dict ={}
for col in cat_cols:
    olympic_data_encode[col] = labelEncoder.fit_transform(olympic_data_encode[col])
 
    le_name_mapping = dict(zip(labelEncoder.classes_,
                        labelEncoder.transform(labelEncoder.classes_)))
 
    mapping_dict[col]= le_name_mapping
print(mapping_dict)

# import csv
# with open('test.csv', 'w') as f:
#     for key in mapping_dict.keys():
#         f.write("%s,%s\n"%(key,mapping_dict[key]))


# ### Impute missing Age, Height and Weight values
imputer = IterativeImputer()
imputer.fit(olympic_data)
Xtrans = imputer.transform(olympic_data)

# Display imputed values
impute_df = pd.DataFrame(Xtrans, columns = ["sex","season","event_category","region","is_medal","age","height","weight"])
impute_df.head()

# Combine imputed values with original olympic_data dataframe to look at the imputed results
# x_impute_df = impute_df.rename(columns={"age": "age_impute", "height": "height_impute","weight": "weight_impute"})
# combined_df = pd.concat([olympic_data, x_impute_df[["age_impute","height_impute","weight_impute"]]], axis=1)
# combined_df.to_csv('impute_df.csv')

# ### Split the data

# Filtering the data for Winter
olympic_winter_enc = impute_df.loc[impute_df["season"]==1]

Xe_winter = (olympic_winter_enc.drop(["is_medal","season"], axis=1))

ye_winter = np.array(olympic_winter_enc["is_medal"])

Xew_train, Xew_test, yew_train, yew_test = train_test_split(Xe_winter, ye_winter, test_size=0.2, random_state=42)

# Filtering the data for Summer
olympic_summer_enc = impute_df.loc[impute_df["season"]==0]

Xe_summer = (olympic_summer_enc.drop(["is_medal","season"], axis=1))
ye_summer = np.array(olympic_summer_enc["is_medal"])

Xes_train, Xes_test, yes_train, yes_test = train_test_split(Xe_summer, ye_summer, test_size=0.2, random_state=42)


# ### Easy Ensemble on Winter data

# Fit the model
eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
eec.fit(Xew_train, yew_train)

# Display the confusion matrix
predictions = eec.predict(Xew_test)
cm = confusion_matrix(yew_test, predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

# Print the imbalanced classification report
print("Confusion Matrix for Winter Model")
print(cm_df)
print("Classification Report for Winter Model")
print(classification_report(yew_test, predictions))

# ### Easy Ensemble on Summer data

# Fit the model
eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
eec.fit(Xes_train, yes_train)

# Display the confusion matrix
predictions = eec.predict(Xes_test)
cm = confusion_matrix(yes_test, predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

# Print the imbalanced classification report
print("Confusion Matrix for Summer Model")
print(cm_df)
print("Classification Report for Summer Model")
print(classification_report(yes_test, predictions))

