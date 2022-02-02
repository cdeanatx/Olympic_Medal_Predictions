

# Olympic Medal Predictions - EDA and Visualizations

## Overview

This section of the Olympic Medal Predictions project is meant to set up the Machine Learning and Web Development teams for success. Below is a full accounting of the original dataset (found [here](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results)) and the steps that were taken to prepare it for the Machine Learning team.

## Exploratory Data Analysis

##### Original Dataset

The original dataset contained **271,116** rows of data, each representing an athlete participating in a specific event in a specific year/season of the Olympic Games, dating back to 1896.

The columns contained were ID, Name, Sex, Age, Height, Weight, Team, NOC, Games, Year, Season, City, Sport, Event, Region, Notes 

##### Dropped Columns

The following columns were immediately dropped from the dataset.

- **Name** - Irrelevant to a machine learning model
- **Team** - The data in this column was mostly blank and was not indicative of the athlete's nationality, sport/event, etc.
- **NOC** - This was the 3 letter abbreviation of the athlete's Olympic Region. We decided to use the full region name instead.
- **Games** - This was a concatenation of year and season. We used both year and season in our model, so this was superflous.
- **City** - The city in which the Olympic Games for the year and season was held. Not relevant to the model.
- **Notes** - The data in this column was only present in ~2% of the rows and was not predictive.

### Cleaning the Data

As previously mentioned, the dataset contained 270k rows stretching back to the 1896 Olympic Games in Athens, Greece. Unfortunately, the predictive biometric data columns (age, height, weight) were mostly missing for those early Olympic Games. Since there were so few values present in those early games, any imputations we would have done would not have been reliable and we determined that it was better to decide upon a "cutoff" year, in which we dropped all rows of the dataset ocurring prior to that year.

We also needed to resolve an issue occurring between the **Sport** and **Event** columns. Essentially, sport was too broad of a category to use as a predictor (the **Athletics** sport alone contained 51 events) and event was too granular (486 unique events since 1964). We ultimately decided to create a new column in the dataset called **Event Category** which was a sort of combination of the **Sport** and **Event** columns. This process is explained in further detail below.

##### Year "Cutoff"

Extensive analysis was done to determine the best year to use as a cutoff. We went about this by comparing the **aggregate total percentage of missing values** for the biometric columns (age, height, weight) vs. the **aggregate percentage of total values in the dataset.** The plot below is a good visual representation of the decision-making process, which we used to ultimately cut any data prior to **1964** out of the dataset.

![Analyze_Missing_Values](https://github.com/cdeanatx/Olympic_Medal_Predictions/blob/main/exploratory_data_analysis/plots/plot_running_na_percent.png)

##### Determining Event Categories

Since our machine learning model was to be mostly based on biometric data, we determined that grouping events from the same sport based on summary statistics of that biometric data was the best way to group events. Also, we determined that most team sports, like **Baseball** should be dropped from the dataset because the variance of biometric data was so large. Click [here](https://github.com/cdeanatx/Olympic_Medal_Predictions/blob/main/exploratory_data_analysis/plots/sport_bio_boxplots.pdf) to see a visual representation of the decision-making process we used to determine event category for sports whose biometric data was too varied to map directly to event category. In the end, we were able to map 468 unique events down to **52** event categories.