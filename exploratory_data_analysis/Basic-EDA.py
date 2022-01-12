import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

olympic_data = "data.csv"

# Step 1: Review the dataset
olympic_df = pd.read_csv(olympic_data)
olympic_df.head()

# Step 1: Review the dataset
olympic_df.tail()

# Step 1: Review the dataset
olympic_df.shape

# Step 2: Understand the dataset, unique values, and missing values
olympic_df.describe()

# Step 2: Understand the dataset, unique values, and missing values
olympic_df.nunique()

# Step 2: Understand the dataset, unique values, and missing values
olympic_df['medal'].unique()

# Step 2: Understand the dataset, unique values, and missing values
olympic_df['region'].unique()

# Step 2: Understand the dataset, unique values, and missing values
olympic_df['year'].unique()

# Step 2: Understand the dataset, unique values, and missing values
olympic_df.isnull().sum()

# Step 3: Cleaning the data
olympic2_df = olympic_df.drop('noc', axis =1)
olympic2_df.head()

# Step 4: Relationship Analysis
corelation = olympic2_df.corr()
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)

# Step 4: Relationship Analysis
olympic2_df.dtypes

# Step 4: Relationship Analysis
sns.pairplot(olympic2_df)

# Step 4: Relationship Analysis
sns.relplot(x= 'height', y='weight', hue='sex', data = olympic2_df)

# Step 4: Relationship Analysis
sns.relplot(x= 'height', y='weight', hue='year', data = olympic2_df)

# Step 4: Relationship Analysis
sns.relplot(x= 'height', y='weight', hue='season', data = olympic2_df)

# Step 5: Sports/Event Analysis
sports_df = olympic2_df.groupby(['sport'])['event','id','year'].nunique()
sports_df

# Step 5: Sports/Event Analysis
athletics_filter = (olympic2_df['sport'] == 'Athletics')
athletics_df = olympic2_df[athletics_filter]
athletics_df

# Step 5: Sports/Event Analysis
athletics2_df = athletics_df.groupby(['event'])['id','year'].nunique()
athletics2_df

# Step 5: Athletics/Male Height-Weight by Event
mens_athletics_filter = (athletics_df['sex'] == 'M')
mens_athletics_df = athletics_df[mens_athletics_filter]

sns.relplot(x= 'height', y='weight', hue='event', data = mens_athletics_df)

# Step 5: Athletics/Female Height-Weight by Event
womens_athletics_filter = (athletics_df['sex'] == 'F')
womens_athletics_df = athletics_df[womens_athletics_filter]

sns.relplot(x= 'height', y='weight', hue='event', data = womens_athletics_df)

# Step 5: Sports/Event Analysis
athletics3_df = athletics_df.groupby(['event'])['age','weight','height'].mean()
athletics3_df

# Step 5: Sports/Event Analysis
swimming_filter = (olympic2_df['sport'] == 'Swimming')
swimming_df = olympic2_df[swimming_filter]
swimming_df

# Step 5: Sports/Event Analysis
swimming2_df = swimming_df.groupby(['event'])['id','year'].nunique()
swimming2_df
swimming3_df = swimming_df.groupby(['event'])['age','weight','height'].mean()
swimming3_df

# Step 5: Swimming/Male Height-Weight by Event
mens_swimming_filter = (swimming_df['sex'] == 'M')
mens_swimming_df = swimming_df[mens_swimming_filter]

sns.relplot(x= 'height', y='weight', hue='event', data = mens_swimming_df)

# Step 5: Swimming/Female Height-Weight by Event
womens_swimming_filter = (swimming_df['sex'] == 'F')
womens_swimming_df = swimming_df[womens_swimming_filter]

sns.relplot(x= 'height', y='weight', hue='event', data = womens_swimming_df)