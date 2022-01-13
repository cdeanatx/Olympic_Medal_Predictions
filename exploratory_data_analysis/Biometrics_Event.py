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

olympic2_df[olympic2_df.sport=="Swimming"].groupby(["sex", "event"])["height", "weight", "age"].mean()

olympic2_df[olympic2_df.sport=="Shooting"].groupby(["sex", "event"])["height", "weight", "age"].mean()

olympic2_df[olympic2_df.sport=="Cycling"].groupby(["sex", "event"])["height", "weight", "age"].mean()
