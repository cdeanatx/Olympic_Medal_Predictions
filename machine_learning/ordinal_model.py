#Ordinal Model code


#imports
import pandas as pd
import scipy.stats as stats
from pandas.api.types import CategoricalDtype
from statsmodels.miscmodels.ordinal_model import OrderedModel

#read csv and split into summer and winter

df = pd.read_csv("final_olympic_data.csv")
summer_df = df[df["season"] == "Summer"]
winter_df = df[df["season"] == "Winter"]

#fill NaNs for medals and create ordered column of medal types

summer_df.medal = summer_df.medal.fillna('None')
winter_df.medal = winter_df.medal.fillna('None')
cat_type = CategoricalDtype(categories=["Gold", "Silver", "Bronze", "None"], ordered=True)
summer_df["medals"] = summer_df['medal'].astype(cat_type)
winter_df["medals"] = winter_df['medal'].astype(cat_type)

#Summer model

summer_model = OrderedModel.from_formula("medals ~ age + height + weight + C(event_category) + C(sex) + C(region)", summer_df,
                                      distr='logit')
summer_fit = summer_model.fit(method='bfgs', maxiter=1000)
summer_fit.summary()

#Winter model

winter_model = OrderedModel.from_formula("medals ~ age + height + weight + C(event_category) + C(sex) + C(region)", winter_df,
                                      distr='logit')
winter_fit = summer_model.fit(method='bfgs', maxiter=1000)
winter_fit.summary()

#Format to get results
#place appropriate information into dataframe
#feed dataframe to summmer_fit or winter_fit predict function

data2 = {'age':[24, 36, 25, 26],
        'weight':[49, 88, 78, 63],
        'height':[165, 195, 183,163],
        'event_category':['gymnastics','swimming', 'archery', 'archery'],
        'sex':['F', 'M', 'F', 'F'],
        'region':['China', 'USA', 'Germany', 'USA']}
test_df = pd.DataFrame(data2)

summer_fit.predict(test_df.iloc[:4])