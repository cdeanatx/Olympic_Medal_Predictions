# Olympic Medal Predictions

## Overview:

### Model 1: Medal prediction

Our goal is to create a machine learning model to predict if a person would win a medal based on the bio-metric data (age, height, weight) they enter. 

### Model 2: Medal probability prediction

In this model, we are predicting probabilities of medaling in various Olympic sports, based on biometric data (age, height, weight) they enter.

## Resources: 
Python 3.7.3, Jupyter Notebook 6.0.0, Scikit learn 1.0.0, statsmodels 0.13.0

## Approach:

As few people on our team are working on cleaning the data, we started with analyzing the data to create a machine learning model. Our approach to build a machine learning model is:

1. Remove the features which doesn’t add much value to the model to reduce overfitting and make it less prone to errors.
	
2. Impute the null values in age, height and weight columns using iterative imputer.
	
3. Use label encoder to encode the categorical features.
	
4. Use EasyEnsembleClassifier model because we are working with unbalanced data. The records with medal counts are far less than the records with no medals. In ensemble method the balance is achieved by randomly under sample.
	
5. Create two models for Summer and Winter Olympics because the events or event categories in both the games are different.

## Summary:

Our team has worked with multiple models such as RandomForest, DecisionTree, … , and EnsembleClassifier but we have chosen the Ensemble classifier because it is addressing the data imbalance in our dataset.

Using an EasyEnsembleClassifier we were able to achieve recall of 69% for winter model and 64% for summer model.


### Ordinal Regression Model Summary

Ordinal regression at it's core is a type of regression analysis designed to predict an ordinal value. Ordinal values come in many forms such as marketing survey data (agree to disagree scale), movie ratings (1 to 10) or in this case Olympic medals. The model makes use of the fact that there is an order to each value and the order itself is significant. The ordinal regression model used for this task belongs to the statsmodels package. Our independent variables are the same as the original model; age, weight, height, sex, region, and event category. All variables are the same as the original dataset except for event category which is derived from a mix of sports and events, but it should still have a relationship with the dependent variable that the model can use. Our dependent variable is our medals category which consists of gold, silver, bronze, and no medal.

The advantage of the statsmodel ordered model is it's from formula option, which allows for categorical variables to be handed to the model with no encoding. As two of our categorical variables, region and event category, are quite large this reduces the number of steps it takes for the model to run. Since the ordered model is really just a type of generalized linear model, we can write the formula as

medals ~ age + weight + height + C(sex) + C(region) + C(event_category)

where C() denotes that the variable is a categorical variable. The only disadvantage of this method is it is not possible to pickle the results of a formula based model so we could not create an interactive piece of the website like we were able to for the other model.

Through the predict function included in the statsmodels package, we can have the model predict the probability of a set of given bio-metric data ending up in each medal category. Due to the fact that these events are highly competitive, the highest categorical result will always be no medal, but ranking the gold probabilities of competitors in a given field can give us insight into exactly what makes someone more likely to win gold in a given event category. For instance, if a region has won gold in the prior olympics for that event category, the gold chance of the next competitor from that region is boosted. If we look at the real life context of the olympics, this result makes sense. Many countries invest heavily in a handful of categories and find continued success in them, for example Kosovo has recently found success in winning gold at the Olympics in judo, but from the number of competitors they field in that event versus others it shows that their investment is higher in judo versus other events.

Through this model we can not only make predictions about future competitors but also understand more about how countries choose to compete in the Olympics, which is invaluable insight into the Olympic medalling process.

