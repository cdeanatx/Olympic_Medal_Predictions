import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

olympic_data = "resources/recent_olympic_data.csv"
olympic_df = pd.read_csv(olympic_data)

# Explore general information about the dataset
print(f'Shape: {olympic_df.shape}\n')
print(f'Stats for numeric columns:\n\n{olympic_df.describe()}\n')

# Explore data by columns
print(f'Counts of unique values in each column:\n\n{olympic_df.nunique()}\n')
print(f'List of values present in the "medal" column: {olympic_df["medal"].unique()}\n')
print(f'List of values present in the "region" column:\n\n{(olympic_df["region"].sort_values().unique())}\n')
print(f'List of values present in the "year" column:\n\n{olympic_df["year"].sort_values().unique()}\n')

# Count missing values by column
print(f'Count of missing values by column:\n\n{olympic_df.isnull().sum()}\n')

# Drop region abbreviations
olympic2_df = olympic_df.drop('noc', axis=1)

# How correlated are each of our variables?
corelation = olympic2_df.corr() # Strong correlation (0.8) between height and weight. Everything else uncorrelated.
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)
plt.savefig('exploratory_data_analysis/plots/correlation_heatmap.png')

sns.pairplot(olympic2_df.drop('id', axis=1))
plt.savefig('exploratory_data_analysis/plots/correlation_pairplot.png')

# What are the data types in each column?
print(f'Column datatypes:\n\n{olympic2_df.dtypes}\n')

# How are height/weight related to sex, year, and season?
sns.relplot(x= 'height', y='weight', hue='sex', data = olympic2_df)
plt.savefig('exploratory_data_analysis/plots/height_weight_by_sex.png')

sns.relplot(x= 'height', y='weight', hue='year', data = olympic2_df)
plt.savefig('exploratory_data_analysis/plots/height_weight_by_year.png')

sns.relplot(x= 'height', y='weight', hue='season', data = olympic2_df)
plt.savefig('exploratory_data_analysis/plots/height_weight_by_season.png')

# Count unique events, participants, and years by sport
sports_df = olympic2_df.groupby(['sport'])[['event','id','year']].nunique().rename(columns={'event': 'n_events', 'id': 'n_participants', 'year': 'n_years'})
print(f'Unique counts by sport:\n\n{sports_df}\n')

# Count unique participants and years by event, where sport is 'Athletics'
athletics_df = olympic2_df[olympic2_df['sport'] == 'Athletics']
athletics2_df = athletics_df.groupby(['event'])[['id','year']].nunique().rename(columns={'id': 'n_participants', 'year': 'n_years'})
print(f'Unique counts by event, where sport is "Athletics":\n\n{athletics2_df}\n')

# Filter athletics df for male/female and assess height/weight by event
mens_athletics_df = athletics_df[athletics_df['sex'] == 'M']
sns.relplot(x= 'height', y='weight', hue='event', data = mens_athletics_df)
plt.savefig('exploratory_data_analysis/plots/male_athletics_height_weight_by_event.png')

womens_athletics_df = athletics_df[athletics_df['sex'] == 'F']
sns.relplot(x= 'height', y='weight', hue='event', data = womens_athletics_df)
plt.savefig('exploratory_data_analysis/plots/female_athletics_height_weight_by_event.png')

# Average age, weight, height by event, where sport is 'Athletics'
athletics3_df = athletics_df.groupby(['event'])[['age','weight','height']].mean()
print('Average age, weight, and height by event:\n\n{athletics3_df}\n')

# Count unique participants and years by event, where sport is 'Swimming'
swimming_df = olympic2_df[olympic2_df['sport'] == 'Swimming']
swimming2_df = swimming_df.groupby(['event'])[['id','year']].nunique().rename(columns={'id': 'n_participants', 'year':'n_years'})
print(f'Unique counts by event, where sport is "Swimming":\n\n{swimming2_df}\n')

# Average age, weight, height by event, where sport is 'Swimming'
swimming3_df = swimming_df.groupby(['event'])[['age','weight','height']].mean()
print(f'Avg age, weight, height by event, where sport is "Swimming":\n\n{swimming3_df}\n')

# Filter swimming df for male/female and assess height/weight by event
mens_swimming_df = swimming_df[swimming_df['sex'] == 'M']
sns.relplot(x= 'height', y='weight', hue='event', data = mens_swimming_df)
plt.savefig('exploratory_data_analysis/plots/male_swimming_height_weight_by_event.png')

womens_swimming_df = swimming_df[swimming_df['sex'] == 'F']
sns.relplot(x= 'height', y='weight', hue='event', data = womens_swimming_df)
plt.savefig('exploratory_data_analysis/plots/female_swimming_height_weight_by_event.png')

# Average height, weight, age by sport/event
olympic2_df.groupby(["sport","event"])["height", "weight", "age"].mean()
