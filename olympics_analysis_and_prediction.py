# Importing the necessary libraries
import warnings
warnings.simplefilter('ignore')
import pandas as pd
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

# Getting data Stats
data_stats = olympic_data.describe()

# Getting the null count columnwise
column_null_counts = olympic_data.isnull().sum(axis=0)

# Filling the NA Values in Medal to 'No Medal'
olympic_data['medal'] = olympic_data['medal'].fillna(value='No Medal')
olympic_data.head()

olympic_data = olympic_data.dropna()
olympic_data.shape

# Choosing the top five countries with the most data
countries = ['USA','BRA','GER','AUS','FRA']
sports = ['Athletics']

# Filtering on the most important features
olympic_data = olympic_data[olympic_data['noc'].isin(countries)]
olympic_data = olympic_data[olympic_data['season'] == 'Summer']
olympic_data = olympic_data[olympic_data['height'].notna()]
olympic_data = olympic_data[olympic_data['age'].notna()]
olympic_data['height (m)'] = olympic_data['height']/100
olympic_data = olympic_data[olympic_data['weight'].notna()] 
olympic_data['bmi'] = round(olympic_data['weight']/(olympic_data['height (m)']*olympic_data['height (m)']), 2)
# olympic_data = olympic_data[olympics['Medal'] == 'Gold']
# wins = ['Gold','Bronze']
# olympic_data = olympic_data[olympic_data['Medal'].isin(wins)]

# Assign "Medal" to the Medal column if the athlete got Bronze, Silver, or Gold. Otherwise, assign "Non-Medal"
olympic_data.loc[(olympic_data['medal'] == 'Gold'),'medal'] = 'Medal'
olympic_data.loc[(olympic_data['medal'] == 'Silver'),'medal'] = 'Medal'
olympic_data.loc[(olympic_data['medal'] == 'Bronze'),'medal'] = 'Medal'
olympic_data.loc[(olympic_data['medal'].isna()),'medal'] = 'Non-Medal'

# Split the data on year = 2016
olympics_df = olympic_data[olympic_data['year'] < 2016]
olympics_2016 = olympic_data[olympic_data['year'] == 2016]

# Using features with the highest importance
X = pd.get_dummies(olympics_df[["sex", "age", "height", "weight", "bmi", "noc"]])
# X = olympics[["BMI","NOC"]]
y = olympics_df["medal"]
print(X.shape, y.shape)
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_2016 = pd.get_dummies(olympics_2016[["sex", "age", "height", "weight", "bmi", "noc"]])
y_2016 = olympics_2016["medal"]

### Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# the closer the one the stronger the coorelation/prediction model
print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")

#Machine learning model predictions tested on 2016 data
# using only the 2016 Olympics dataset to see how close our predictions are
predictions = classifier.predict(X_2016)
print(f"First 10 Predictions: {predictions[:10]}")
print(f"First 10 Actual labels: {y_2016[:10].tolist()}")

# putting actuals and predictions into a dataframe
Testing = pd.DataFrame({"Prediction": predictions, "Actual": y_2016}).reset_index(drop=True)

#Reviewing prediction data
Testing_crosstab = pd.crosstab(Testing['Actual'],Testing['Prediction'])
Testing[(Testing['Prediction'] == 'Medal')]

## Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Calculating the accuracy score.
rf_predictions = rf_model.predict(X_test)
acc_score = accuracy_score(y_test, rf_predictions)

# Calculating the confusion matrix.
cm = confusion_matrix(y_test, rf_predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
print("Confusion Matrix")
print(cm_df)
print(f"Accuracy Score : {acc_score}")
print("Classification Report")
print(classification_report(y_test, rf_predictions))

#Sort the features by their importance.6`
sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)

#Machine learning model predictions tested on 2016 data
# using only the 2016 Olympics dataset to see how close our predictions are
rf_predictions = rf_model.predict(X_2016)

# putting actuals and predictions into a dataframe
Testing = pd.DataFrame({"Prediction": rf_predictions, "Actual": y_2016}).reset_index(drop=True)
