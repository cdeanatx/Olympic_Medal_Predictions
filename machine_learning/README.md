# Olympic Medal Predictions

## Overview:
### Model1: Medal prediction
Our goal is to create a machine learning model to predict if a person would win a medal based on the bio-metric data  (age, height, weight) they enter. 

### Model2: Medal probability prediction
This purpose of this repository is to predict probabilities of medaling in various Olympic sports, based on biometric data.

## Resources: 
Python 3.7.3, Jupyter Notebook 6.0.0, Scikit learn 1.0.0

## Approach:

As few people on our team are working on cleaning the data, we started with analyzing the data to create a machine learning model. Our approach to build a machine learning model is:
	1. Removed the features which doesn’t add much value to the model to reduce overfitting and make it less prone to errors.
	2. We are imputing the null values in age, height and weight columns using iterative imputer.
	3. We are using label encoder to encode the categorical features.
	4. We are using EasyEnsembleClassifier model because we are working with unbalanced data. The records with medal counts are far less than the records with no medals. In ensemble method the balance is achieved by randomly under sample.
	5. We have created two models for Summer and Winter Olympics because the events or event categories in both the games are different.

## Summary:
Our team has worked with multiple models such as RandomForest, DecisionTree, … , and EnsembleClassifier but we have chosen the Ensemble classifier because it is addressing the data imbalance in out dataset.
Using an EasyEnsembleClassifier we were able to achieve recall of 69% for winter model and 64% for summer model.
