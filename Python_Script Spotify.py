#!/usr/bin/env python
# coding: utf-8

# In[2]:


__author__ = 'Phil Baltazar'
__email__  = 'phillusnow@gmail.com'
__website__= 'www.github.com/pbswe'


# # 1. Define 
# The Problem.

# #### A small dataset of the top Spotify songs from the last decade, measured by popularity. 
# 
# We would like to know if there is any relation between these top songs and their genre, or artist, or length or any other feature present in this dataset that show us correlation with popularity, or what traits (if any) do popular songs have to each other. 

# ###### Note: This dataset was downloaded from Kaggle at
# https://www.kaggle.com/leonardopena/top-spotify-songs-from-20102019-by-year

# In[177]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing, metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV,RepeatedStratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
import inspect
import xgboost
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')


# # 2. Discover
# The data.

# #### -- Loading the data --

# In[196]:


df = pd.read_csv('../Spotify/top10s.csv')
df50 = pd.read_csv('../Spotify/top50.csv')


# In[84]:


# Visualize the data
df.head()


# #### -- Cleaning the data --

# In[85]:


df.columns


# In[86]:


# Adjusting column/header titles
header = ['Index', 'Title', 'Artist', 'Genre', 'Year', 'BPM',
       'Energy', 'Danceability', 'Loudness dB', 'Liveness', 'Valence',
       'Length', 'Acousticness', 'Speechiness', 'Popularity']
df.columns = header


# In[87]:


df = df.drop([0], axis=0)


# In[88]:


df.head()


# In[200]:


# Adjusting column/header titles for df50 and types.
header50 = ['Index', 'Title', 'Artist', 'Genre', 'Year', 'BPM', 'Energy', 'Danceability', 
            'Loudness dB', 'Liveness', 'Valence', 'Length', 'Acousticness', 
            'Speechiness', 'Popularity']
df50.columns = header50
df50[['Title', 'Artist', 'Genre']] = df50[['Title', 'Artist', 'Genre']].astype('category')
df50.info()


# In[89]:


# Checking data types
df.info()


# In[90]:


# Fixing data types. Not all numerical data should be handled as numbers. 
# For example, popularity is shown as numbers but should still be categorical. 

# Calling columns again after fixed:

df.columns


# In[91]:


categoricCols = ['Index', 'Title', 'Artist', 'Genre', 'Popularity']
numericCols = ['Year', 'BPM', 'Energy', 'Danceability', 'Loudness dB',
                 'Liveness', 'Valence', 'Length', 'Acousticness', 'Speechiness']

df[categoricCols] = df[categoricCols].astype('category')
df[numericCols] = df[numericCols].astype('int')


# In[92]:


# Checking dtypes again to see if they were changed correctly.
df.info()


# In[93]:


# Checking for duplicates (zero is good, zero means there aren't any duplicates).
df.duplicated().sum()


# In[94]:


df.describe()


# #### -- Exploring the data (EDA) --

# In[95]:


df['Popularity'].value_counts()


# In[96]:


# Exploring upper and lower quartiles (IQR)
# Will require Popularity dtype change to 'int'.

df['Popularity'] = df['Popularity'].astype('int')

dataset = df.Popularity.describe()
print(dataset)
IQR = dataset['75%']-dataset['25%']
upperlimit = dataset['75%'] + (1.5*IQR)
lowerlimit = dataset['25%'] - (1.5*IQR)
print("The upper outlier limit is", upperlimit)
print("The lower outlier limit is", lowerlimit)


# In[97]:


# Explore the target variable: popularity.
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
sns.boxplot(df.Popularity)
plt.subplot(1, 2, 2)
sns.distplot(df.Popularity, bins=5)
plt.show()


# In[98]:


# Creating a function to plot the feature variables.
# This will help visualize possible correlation with popularity. 
def plotFeats(df, var): 
    '''
    creates a plot for each feature
    plot1(left), sample distribution
    plot2(right), popularity dependance/relationship
    '''
    plt.figure(figsize = (30, 15))
    plt.subplot(1, 2, 1)
    if df[var].dtype == 'int':
        plt.hist(df[var], bins=5)
    else:
        #change the object datatype of each variable to category type and \n
        #order their level by the mean popularity
        mean = df.groupby(var)['Popularity'].mean()
        df[var] = df[var].astype('category')
        level = mean.sort_values().index.tolist()
        df[var].cat.reorder_categories(level, inplace=True)
        df[var].value_counts().plot(kind='bar')   
    plt.xticks(rotation=45, size=8)
    plt.xlabel(var)
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)

    if df[var].dtype == 'int': 
       #Plot the mean popularity for each category and shade the line \n 
       #between the (mean - std, mean + std)
        mean = df.groupby(var)['Popularity'].mean()
        std = df.groupby(var)['Popularity'].std()
        mean.plot()
        plt.fill_between(range(len(std.index)), mean.values-std.values, mean.values + std.values,alpha = 0.1)
    else:
        sns.boxplot(x= var, y='Popularity', data= df)
    
    plt.xticks(rotation=45)
    plt.ylabel('Popularity')
    plt.show()
#    plt.savefig("image.png",bbox_inches='tight',dpi=500)


# In[99]:


# Plotting every feature against "Popularity", using the function above.
# Starting with categorical values, followed by the numerical values after.


# In[100]:


plotFeats(df, 'Artist')


# In[101]:


plotFeats(df, 'Genre')


# In[102]:


plotFeats(df, 'Year')


# In[103]:


plotFeats(df, 'BPM')


# In[104]:


plotFeats(df, 'Energy')


# In[105]:


plotFeats(df, 'Danceability')


# In[106]:


plotFeats(df, 'Loudness dB')


# In[107]:


plotFeats(df, 'Liveness')


# In[108]:


plotFeats(df, 'Valence')


# In[109]:


plotFeats(df, 'Length')


# In[110]:


plotFeats(df, 'Acousticness')


# In[111]:


plotFeats(df, 'Speechiness')


# ###### Our plotted models reveal an interesting insight from our data. 
# 
# <i>It turns out that many features have a very poor relation with popularity. Because the relationship is poor, this means that these particular features are not correlated with popularity. In other words, these features, which we'll explain below, do not correlate with the popularity of that particular song. There were a few exceptions though... </i>
# 
# <br>
# Starting from the beginning, let's look at the 3 best features:
# 
# <h4>"Artist"</h4>
# 
# The Artist plot shows a visible sine wave across the popularity of each song. 
# 
# <b>"Does this mean a particular artist has higher chances of releasing a top hit just from being who they are?"</b>
# 
# The answer is a bit more complicated than a simple 'yes' or 'no'. We will need to compare artists against other features to gain more insight... (More visualization on Tableau). But in short, yes, this could be an indicator. 
# 
# <br>
# Moving on, second on our list is:
# <h4>"Genre"</h4> 
# As it has an almost linear correlation with popularity. What does this mean?
# 
# Depending on the genre, the likelihood of hitting the top of the charts increases. This doesn't assume that this dataset could simply have more songs of that particular genre, but if believed to be balanced, genre has a strong correlation to popularity. 
# 
# <br>
# The last one that is worth mentioning:
# <h4>Loudness db</h4>
# 
# There seems to be, again, a linear correlation to loudness against popularity. Here's an insider insight: songs that are professionally and strongly mastered are likely to be heard more times. Loudness here means how clear you can hear that song in a set volume as you play the entire playlist. Each song competes for attention and the louder the sound is without distorting, the better each element will sound, which in turn will increase the likelihood of the song popularity. 
# 
# To summarize, loudness increases the popularity based on the assumption that the clearer and more potent all the elements in the song are heard, the better the melody will be perceived and therefore liked, resulting in increased popularity. 
# 
# <br>
# All other features seemed to be spreaded throughout the spectrum of popularity, which shows us the correlation is weak. For example, 'length' is insiginificant when prediciting popularity because a short or a long length gives no insight on how popular that song is. 

# In[112]:


# Visualizing all these features in a heatmap.

plt.figure(figsize = (12, 8))
corr = df.corr()
sns.heatmap(corr, 
           xticklabels = corr.columns.values,
           yticklabels = corr.columns.values,
           cmap = "magma", vmin = - 1, vmax = 1, annot = True, linewidths = 1)
plt.title('Heatmap of Correlations Matrix')
plt.xticks(rotation=45)
plt.show()


# Interestingly enough, the heatmap not only shows the correlation with popularity, but with all other features against one another. Acousticness vs. Energy seems to have a reasonable correlation.
# 
# Note: Artist and Genre aren't shown in the heatmap because they aren't ordinal, but we saw on the boxplots above that they're the strongest correlators to popularity.

# # Now, we ask the question:
# How likely is a song to hit top popularity based on its artist, genre and loudness?
# <br>
# <br>
# Where:<br>
# "Artist, Genre and Loudness" were determined to be the strongest correlators to popularity. 
# 
# <br>
# "Top Popularity" is a popularity score within the 75% upper percentile.
# <br>
# <br>
# "Likelihood" assesses the chances a song has of meeting the above criteria based on those attributes. 
# 
# Meaning, does song "One" from artist A1, genre B1 and loudness C1 have a higher change of hitting 75% or more in popularity than song "One" from artist A2, genre B2 and loudness C2?

# #### Establishing a baseline
# 
# Based on our data, we will use classification methods like Linear SVC (Support Vector Classification) or Naive Bayes to predict the model. Improving and fine tuning the model will come later on.

# In[113]:


def EncodeData(df):
    for col in df.columns:
        if df[col].dtype.name == 'category':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    return df


# In[114]:


baselineDF = df.copy()
baselineDF = EncodeData(baselineDF)


# In[115]:


featsBaseline = baselineDF[['Index', 'Title', 'Artist', 'Genre', 'Year', 
                            'BPM', 'Energy', 'Danceability', 'Loudness dB', 
                            'Liveness', 'Valence', 'Length', 'Acousticness', 
                            'Speechiness', 'Popularity']]
targsBaseline = baselineDF[['Popularity']]


# In[116]:


lr = LinearRegression()
baseLrMse = cross_val_score(lr, featsBaseline, targsBaseline, scoring = 'neg_mean_squared_error')
baseLrMse = -1*baseLrMse.mean()
print('Baseline MSE Score: ', baseLrMse)


# #### Hypothesizing solutions
# 
# Choosing some 3 models that may improve results over the baseline model.

# # 3. Develop
# The model.

# 
# Now we'll create features, tune models and train/valide them until we reach an efficacy goal.

# #### Feature Engineering
# Using OHE (one-hot-encoding) to transform categorical features into data types we can manipulate and make calculations with. 

# In[117]:


# Dropping 'index' because it is useless for our OHE algorithms. 
df = df.drop(['Index'], axis=1)


# In[118]:


# Check to see that the column was dropped with: 
df.info()
#or
#df.columns


# In[119]:


categoryDF = df[['Title', 'Artist', 'Genre']]
categoryDF = pd.get_dummies(categoryDF, drop_first=True)


# In[120]:


# Normalizing values.

normalDF = df[['Year', 'BPM', 'Energy', 'Danceability', 'Loudness dB', 'Liveness', 
               'Valence', 'Length', 'Acousticness', 'Speechiness']]
cols = normalDF.columns
normalDF = MinMaxScaler().fit_transform(normalDF)
normalDF = pd.DataFrame(normalDF, columns = cols)


# In[121]:


# Concatenating both into one dataframe.

featsDF = pd.concat([categoryDF.reset_index(drop=True), normalDF], axis=1)
targsDF = df[['Popularity']]


# In[122]:


featsDF.shape


# In[123]:


del categoryDF, normalDF


# #### Creating Models

# In[124]:


# Out of the models we brainstormed earlier, we'll create and fine tune them now. 
# Starting with a kFold - Cross Validation, where k = 5.

def evalModel(model):
    negMse = cross_val_score(model, featsDF, targsDF.values.ravel(), 
                             scoring = 'neg_mean_squared_error')
    mse = -1 * negMse
    stdMse = round(mse.std(), 2)
    meanMse = round(mse.mean(), 2)
    print('\nModel\n', model)
    print('    Standard Deviation of Cross Validation MSEs:\n    ', stdMse)
    print('    Mean 5-Fold Cross Validation MSE: \n    ', meanMse)
    return meanMse


# #### Testing Models

# In[125]:


models = []
meanMse = {}

# Linear Regression
lr = LinearRegression()

# Stochastic Gradient Descent
sgd = SGDRegressor(max_iter=1000, learning_rate='optimal')

# Decision Tree Regressor
dtr = DecisionTreeRegressor(max_depth=15)

# Random Forest Regressor
rfr = RandomForestRegressor(n_estimators=150, n_jobs=-1, max_depth=30, 
                            min_samples_split=60, max_features='sqrt')

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(max_depth = 3, n_estimators=150, learning_rate = 0.1, 
                                 n_iter_no_change=10)

# Extreme Gradient Boosting
xgb = xgboost.XGBRegressor(max_depth=5, n_estimators=150, learning_rate=0.1, n_jobs=-1)

models.extend([lr, sgd, dtr, rfr, gbc, xgb])
print('Cross Validation of Models Initiated...\n')

for model in models:
    mseIter = evalModel(model)
    meanMse.update({model:mseIter})
    
bestModel = min(meanMse, key=meanMse.get)

print('\n\nThe model with the lowest average MSE to use for predictions is:\n')
print(bestModel)


# In[126]:


# Adding the meanMse dictionary into a DataFrame.

modelsDF = pd.DataFrame.from_dict(data = meanMse, orient='index', columns=['MSE-Score'])
modelsDF.index = ['LR', 'SDG', 'DTR', 'RF', 'GBC', 'XGB']
modelsDF


# #### Selecting the best model

# Based on the results above, Random Forest performed better than the others. We will use RF to fine tune it and test it. 

# In[127]:


trainX, testX, trainY, testY = train_test_split(featsDF, targsDF, random_state=36, test_size=0.2)


# In[128]:


# Initiating array to store results
results = []

# Initiating a watchlist to keep track of performance
evalSet = [(trainX, trainY), (testX, testY)]

# Checking hyperparameters.
print(inspect.signature(RandomForestRegressor))


# In[129]:


results = []

for n_estimators in [100, 150, 250, 500, 750, 1000, 1500]: #, 2000, 3000, 5000, 10000]:
    clf = RandomForestRegressor(n_estimators = n_estimators, n_jobs = -1, max_depth = 100,
                               min_samples_split = 60, max_features = 'sqrt')
    clf.fit(trainX, trainY)
    results.append(
        {
            'n_estimators': n_estimators,
            'train_error': metrics.mean_squared_error(trainY, clf.predict(trainX)),
            'test_error': metrics.mean_squared_error(testY, clf.predict(testX))
        })
    
# Showing results
nEstimatorsLr = pd.DataFrame(results).set_index('n_estimators').sort_index()
nEstimatorsLr


# In[130]:


# Let's plot those results above.

nEstimatorsLr.plot(title='nEstimator Learning Curve')


# The results show that the model remains constant throughout with a slight improvement at 500, so that will be our nEstimator hyperparameter value.

# In[132]:


results = []

for max_depth in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 500]:
    clf = RandomForestRegressor(max_depth = max_depth, max_features='sqrt',
                                n_estimators = 500, n_jobs = -1)
    clf.fit(trainX, trainY)
    results.append(
        {
            'max_depth': max_depth,
            'train_error': metrics.mean_squared_error(trainY, clf.predict(trainX)),
            'test_error': metrics.mean_squared_error(testY, clf.predict(testX))
        })
    
# Displaying the results.
maxDepthLr = pd.DataFrame(results).set_index('max_depth').sort_index()
maxDepthLr


# In[133]:


# Visualizing the maxDepth learning curve.
maxDepthLr.plot(title='maxDepth Learning Curve')


# Train stops improving at 70. Test is also one of the lowest at 80, so that will be our maxDepth hyperparameter value.

# In[134]:


results = []

for min_samples_split in [2, 4, 8, 20, 40, 60, 80, 100, 200, 500]:
    clf = RandomForestRegressor(max_depth=80, max_features='sqrt', 
                                n_estimators=500, n_jobs=-1)
    clf.fit(trainX, trainY)
    results.append(
        {
            'min_samples_split': min_samples_split,
            'train_error': metrics.mean_squared_error(trainY, clf.predict(trainX)),
            'test_error': metrics.mean_squared_error(testY, clf.predict(testX))
        })
    
    
    
# Displaying Results  
minSamples = pd.DataFrame(results).set_index('min_samples_split').sort_index()
minSamples


# In[135]:


# Visualizing the minSamples learning curve. 
minSamples.plot(title='minSamples Learning Curve')


# At 20, both train and test are at their best so that will be the value for min_samples_split hyperparameter.

# In[136]:


# Saving the best hyperparameter setting.
bestModel = RandomForestRegressor(max_depth=70, n_estimators=500, min_samples_split=80, 
                                  criterion='mse', n_jobs=-1, 
                                  verbose=2, min_impurity_decrease = 1.0)


# In[137]:


# Fitting the training set and applying early stopping.
bestModel.fit(trainX, trainY)


# In[138]:


# Saving the best hyperparameter setting with n_estimators from early stopping.
bestModel = RandomForestRegressor(max_depth=70, max_features='sqrt', min_samples_split=20, 
                                  n_estimators=300, n_jobs=-1)


# In[139]:


evalModel(bestModel)


# #### Conclusion
# 
# Fine tuning improved the model a little bit (from 195.xx to 194.xx), so we'll use these tuned hyperparameters to deploy the solution.

# # 4. Deploy
# The solution.

# Automating the pipeline. Writting a function to train the whole training set and saves to disk.

# In[215]:


def trainedCleanModelDF (df, df50):
    
    # Loading csv files
    featsDF = pd.read_csv(df)
    targsDF = pd.read_csv(df50)
    
    # Cleaning feature and target dataframes per analysis above
    df = pd.concat([featsDF, targsDF], axis=0)
    catsDF = df[['Title', 'Artist', 'Genre']]
    catsDF = pd.get_dummies(catsDF, drop_first=True)
    featsDF = pd.concat([catsDF, df[['Year', 'BPM', 'Energy', 'Danceability', 'Loudness dB', 
                                     'Liveness', 'Valence', 'Length', 'Acousticness', 
                                     'Speechiness']]], axis=1)
    targsDF = df[['Popularity']]
    del catsDF #, df
    
    # Implement best model discovered per analysis above
    model = RandomForestRegressor(max_depth=70, max_features='sqrt', min_samples_split=20,
                      n_estimators=300, n_jobs=-1)
    model.fit(featsDF, targsDF)
    
    # Save model to disk
    model_file = 'model'
    pickle.dump(model, open(model_file, 'wb'))
    
    # Informs user that process is complete
    print('Data prepraration and model creation complete.')


# In[216]:


# Script that prepares data, predicts popularity, and exports results.

class popularityPredictionModel():
    
    # Read the 'model' file which was saved.
    def __init__(self, model_file):
        self.rfr = pickle.load(open(model_file, 'rb'))
    
    # Takes/prepares data, makes predictions from trained model, and exports results to csv file.
    def exportPredictions(self, data_file):

        # Load csv file.
        df_pred_features = pd.read_csv(data_file)
    
        # Saves Popularity column for output file.
        df_pred_popular = pd.DataFrame(df_pred_features['Popularity'])
        
        # Prepares data to be fed into the model
        df_pred_categories = df_pred_features[['Title', 'Artist', 'Genre', 'Year', 
                                               'BPM', 'Energy', 'Danceability', 
                                               'Loudness dB', 'Liveness', 'Valence', 
                                               'Length', 'Acousticness', 'Speechiness']]
        df_pred_categories = pd.get_dummies(df_pred_categories, drop_first=True)
        df_pred_features = pd.concat([df_pred_categories, df_pred_features], axis=1)
        del df_pred_categories
    
        # Loads model from disk, predicts popularity, and exports results to .csv file
        df_pred = pd.DataFrame(self.rfr.predict(df_pred_features))
        df_pred.columns = ['Popularity']
        df_pred = pd.concat([df_pred_popular,df_pred], axis=1)
        df_pred.to_csv('predicted_popularity.csv')
        del df_pred_popular
        
        # Informs user that process is complete
        print('Predictions exported to .csv file.')
    
    # Plot feature importance of model and save figure to .jpg file
    def exportFeatureImportance(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        RandonForestRegressor.plot_importance(self.rfr, height=0.6, ax=ax)
        fig.savefig('feature_importance.jpg')
    
        # Informs user that process is complete
        print('Feature importances exported to .jpg file.')


# In[222]:


trainedCleanModelDF('top10s.csv', "top50.csv")


# In[223]:


model = popularityPredictionModel('model')


# In[226]:


model.exportPredictions('../Spotify/test_spotify.csv')


# In[228]:


model.exportFeatureImportance()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Work in progress...

