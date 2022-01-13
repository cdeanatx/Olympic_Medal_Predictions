import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

olympic_data = "resources/recent_olympic_data.csv"
olympic_df = pd.read_csv(olympic_data)

olympic_df = pd.read_csv(olympic_data)
olympic_df.head()

sports_df = olympic2_df.groupby(['sport'])['event','id','year'].nunique().sort_values(by='event', ascending=False)
sports_df

olympic2_df = olympic_df.drop('noc', axis =1)
olympic2_df.head()

olympic2_df[olympic2_df.sport=="Athletics"].groupby(["sex", "event"])["height", "weight", "age"].mean()

athletics_filter = (olympic2_df['sport'] == 'Athletics')
athletics_df = olympic2_df[athletics_filter]
athletics_df
mens_athletics_filter = (athletics_df['sex'] == 'M')
mens_athletics_df = athletics_df[mens_athletics_filter]
fig, ax = plt.subplots(figsize=(10, 10))
sns.histplot(data=mens_athletics_df, x='height', hue='event', stat='count', edgecolor=None)
ax.set_title('Mens Athletics Height Dist')

fig, ax = plt.subplots(figsize=(10, 10))
sns.histplot(data=mens_athletics_df, x='weight', hue='event', stat='count', edgecolor=None)
ax.set_title('Mens Athletics Weight Dist')

olympic2_df[olympic2_df.sport=="Swimming"].groupby(["sex", "event"])["height", "weight", "age"].mean()

swimming_filter = (olympic2_df['sport'] == 'Swimming')
swimming_df = olympic2_df[swimming_filter]
swimming_df
mens_swimming_filter = (swimming_df['sex'] == 'M')
mens_swimming_df = swimming_df[mens_swimming_filter]
fig, ax = plt.subplots(figsize=(10, 10))
sns.histplot(data=mens_swimming_df, x='height', hue='event', stat='count', edgecolor=None)
ax.set_title('Mens Swimming Height Dist')

sns.histplot(data=mens_swimming_df, x='weight', hue='event', stat='count', edgecolor=None)
ax.set_title('Mens Swimming Weight Dist')

olympic2_df[olympic2_df.sport=="Shooting"].groupby(["sex", "event"])["height", "weight", "age"].mean()

olympic2_df[olympic2_df.sport=="Cycling"].groupby(["sex", "event"])["height", "weight", "age"].mean() 
