
# Importing the necessary libraries
import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# Reading the dataset
olympic_data = pd.read_csv("cleaned_olympic_data.csv")


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

# 172495 null values
olympic_data["medal"].value_counts()

# Replace medal NA to 0,1
olympic_data["medal"] = olympic_data["medal"].fillna(0)
olympic_data["is_medal"] = np.where(olympic_data["medal"] == 0, 0, 1)
olympic_data

# Null Count
olympic_data.isnull().sum(axis=0)

# Olympic data columns
olympic_data = olympic_data[["id","sex","year","season","event_category","is_medal","age","height","weight"]]

# Print data types
olympic_data.dtypes

# Split the data
X = olympic_data.drop(["is_medal"], axis=1).loc[0:199]
y = np.array(olympic_data["is_medal"])[:200]

# Import libraries
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
#import category_encoders as ce

from sklearn.metrics import precision_recall_fscore_support as score

import warnings
warnings.simplefilter('ignore')
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from numpy import isnan
import numpy as np

from sklearn.metrics import recall_score

# define modeling pipeline
model = RandomForestClassifier()
imputer = IterativeImputer()
pipeline_RF = make_pipeline(OneHotEncoder(sparse=False,handle_unknown='ignore'),
                            imputer, 
                            model)

# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline_RF, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise') 

#recall scoring
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# Fit RF pipeline
pipeline_RF.fit(X,y)
predictions=pipeline_RF.predict(X)
recall = recall_score(y, predictions, average='macro')

# Print Recall
print('Recall: %.3f' % recall)

# Validate the data
validation_size = 0.20
seed = 6

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,y,test_size = validation_size, random_state = seed)

scoring = 'accuracy'

# Pipelines for different models to check performance
imputer = IterativeImputer()
pipeline_RF = make_pipeline(OneHotEncoder(sparse=False,handle_unknown='ignore'),
                            imputer, 
                            model
                          )
pipeline_LR = make_pipeline(
                            OneHotEncoder(sparse=False,handle_unknown='ignore'),
                            imputer,
                            LogisticRegression()
                          )
pipeline_LDA = make_pipeline( 
                             OneHotEncoder(sparse=False,handle_unknown='ignore'),
                                imputer,
                            LinearDiscriminantAnalysis()
                          )
pipeline_KNN = make_pipeline( 
                             OneHotEncoder(sparse=False,handle_unknown='ignore'),
                            imputer,
                            KNeighborsClassifier()
                          )
pipeline_DTC = make_pipeline( 
                             OneHotEncoder(sparse=False,handle_unknown='ignore'),
                            imputer,
                            DecisionTreeClassifier()
                          )
pipeline_GNB = make_pipeline(OneHotEncoder(sparse=False,handle_unknown='ignore'),
                            imputer, 
                            GaussianNB()
                          )
pipeline_SVM = make_pipeline(
                             OneHotEncoder(sparse=False,handle_unknown='ignore'),
                             imputer,
                            SVC()
                          )

pipelines=[("RF",pipeline_RF),("LR",pipeline_LR),("LDA",pipeline_LDA),("KNN",pipeline_KNN),("DTC",pipeline_DTC),("GNB",pipeline_GNB),("SVM",pipeline_SVM)]


results = []
names = []

for name, pipeline in pipelines:
    kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv_results = model_selection.cross_val_score(pipeline, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


predictions=[]
for name,pipeline in pipelines:
    pipeline.fit(X_train,Y_train)
    predictions.append(pipeline.predict(X_test))

recalls=[]
for name,prediciton in predictions:
    recalls.append(recall_score(Y_test, prediction, average='macro'))
    

