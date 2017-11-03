"""
    Project/competition: House price prediction

    The purpose of this script is to enable data preprocessing, data transformation and feature engineering for
    training and testing data sets for a problem to predict house sales prices in UK in 2016 based on training data for
    1995-2015.
"""

# import necessary libraries
import datetime as dt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import warnings
import scipy.stats as stats  # use norm, skew for some statistics

import sklearn.preprocessing as pre # use LabelEncoder
import scipy.special as spec # use boxcox1p
from subprocess import check_output
from geopy.geocoders import Nominatim

import sklearn.ensemble as ens # use RandomForestRegressor,  GradientBoostingRegressor
import scipy.special as spec # use boxcox1p
import lightgbm as lgb  # Microsoft's light gbm package

# The error metric: RMSE on the log of the sale prices.
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, RegressorMixin

################################################################################
class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None):
        self.regressors = regressors

    def fit(self, X, y):
        for regressor in self.regressors:
            regressor.fit(X, y)

    def predict(self, X):
        self.predictions_ = list()
        for regressor in self.regressors:
            self.predictions_.append(np.exp(regressor.predict(X).ravel()))

        return np.log1p(np.mean(self.predictions_, axis=0))


################################################################################
# RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

################################################################################
# Cross-validation
def evaludate_model(model, x, y):
    # print('Cross_validation..')
    n_splits_val = 3
    kf = KFold(n_splits=n_splits_val, shuffle=False)
    idx = 0
    rmse_buf = np.empty(n_splits_val)
    for train, test in kf.split(x):
        model.fit(x.iloc[train], y.iloc[train])
        y_cv = model.predict(x.iloc[test])
        rmse_buf[idx] = rmse(y.iloc[test], y_cv)
        # print('Interation #' + str(idx) + ': RMSE = %.5f' % rmse_buf[idx])
        idx += 1

    mean_rmse = np.mean(rmse_buf)
    print('   Mean RMSE = %.5f' % mean_rmse + ' +/- %.5f' % np.std(rmse_buf))

    return mean_rmse


def evaludate_submodels(models, x, y):
    # print('Cross_validation..')
    n_splits_val = 10
    kf = KFold(n_splits=n_splits_val, shuffle=False)
    for m_i, model in enumerate(models.regressors):
        rmse_buf = np.empty(n_splits_val)
        idx = 0
        for train, test in kf.split(x):
            model.fit(x.iloc[train], y.iloc[train])
            y_cv = model.predict(x.iloc[test])
            rmse_buf[idx] = rmse(y.iloc[test], y_cv)
            # print('Interation #' + str(idx) + ': RMSE = %.5f' % rmse_buf[idx])
            idx += 1

        mean_rmse = np.mean(rmse_buf)
        print('Model #' + str(m_i) + ': mean RMSE = %.5f' % mean_rmse + \
              ' +/- %.5f' % np.std(rmse_buf))


################################################################################


def get_longitude(location_address):
    geolocator = Nominatim()
    # location = geolocator.geocode("175 5th Avenue NYC")
    location = geolocator.geocode(location_address)
    return location.longitude
    # print((location.latitude, location.longitude))

def get_latitude(location_address):
    geolocator = Nominatim()
    # location = geolocator.geocode("175 5th Avenue NYC")
    location = geolocator.geocode(location_address)
    return location.latitude


# Convert categorical features using one-hot encoding.
def onehot(onehot_df, df, column_name, fill_na, drop_name):
    onehot_df[column_name] = df[column_name]
    if fill_na is not None:
        onehot_df[column_name].fillna(fill_na, inplace=True)

    dummies = pd.get_dummies(onehot_df[column_name], prefix='_' + column_name)

    # Dropping one of the columns actually made the results slightly worse.
    if drop_name is not None:
         dummies.drop(['_' + column_name + '_' + drop_name], axis=1, inplace=True)

    onehot_df = onehot_df.join(dummies)
    onehot_df = onehot_df.drop([column_name], axis=1)
    return onehot_df

def munge_onehot(df):
    onehot_df = pd.DataFrame(index=df.index)
    # Old_New, Duration, PPD_Category_Type, Property_Type
    # Town, District, County
    onehot_df = onehot(onehot_df, df, 'Old_New', None, None)
    onehot_df = onehot(onehot_df, df, 'Duration', None, None)
    onehot_df = onehot(onehot_df, df, 'PPD_Category_Type', None, None)
    onehot_df = onehot(onehot_df, df, 'Property_Type', None, None)
    onehot_df = onehot(onehot_df, df, 'Town', None, None)
    onehot_df = onehot(onehot_df, df, 'District', None, None)
    onehot_df = onehot(onehot_df, df, 'County', None, None)

    return onehot_df

def munge_geolocation(df):
    geolocation_df = df.copy()
    strLocationAddress = ""

    latitudes = []
    longitudes = []

    for index, row in geolocation_df.iterrows():
        str_street = row['Street']
        str_town = row['Town']

	# TODO: to be properly implemented and reused in feature engineering


def drop_garbage_cols(df):
    result_df = df.copy()

    # Drop these columns. They are either not very helpful or they cause overfitting.
    drop_cols = [
        'Postcode',
        'Street',
        'Locality' ,
        'Date',
        'Property_Type', 'Old_New', 'Duration', 'Town', 'District', 'County', 'PPD_Category_Type'
    ]

    result_df.drop(drop_cols, axis=1, inplace=True)

    return result_df

############################################################
# MAIN EXECUTION LOOP
############################################################

start_time = dt.datetime.now()
print("Started preprocessing.py at ", start_time)

fname_datafile = "input/training.csv"

# output files to store pre-processing data sets
fname_out_training_preprocessed = "input/train_preprocessed.csv"
fname_out_testing_preprocessed = "input/test_preprocessed.csv"

# internal setup not needed to control via command-line arguments
color = sns.color_palette()
sns.set_style('darkgrid')
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) # Limiting floats output to 3 decimal points
warnings.filterwarnings('ignore') # Supress unnecessary warnings for readability

# Now let's import and put the train and test datasets in  pandas dataframe
all_data = pd.read_csv(fname_datafile)

# get info about the raw data
# RangeIndex: 100000 entries, 0 to 99999
# Data columns (total 13 columns):
# ID                   100000 non-null int64
# Price                98818 non-null float64
# Date                 100000 non-null object
# Postcode             99897 non-null object
# Property_Type        100000 non-null object
# Old_New              100000 non-null object
# Duration             100000 non-null object
# Street               100000 non-null object
# Locality             73176 non-null object
# Town                 100000 non-null object
# District             100000 non-null object
# County               100000 non-null object
# PPD_Category_Type    100000 non-null object
# dtypes: float64(1), int64(1), object(11)
print (all_data.info())


# Add the one-hot encoded categorical features.
onehot_df = munge_onehot(all_data)
all_data = all_data.join(onehot_df)

all_data = drop_garbage_cols(all_data)
print('---------------------------------------------')
print("All_data before splitting into train and test:")
print (all_data.info())

#print(train['Locality'].value_counts())

# split all_data into training and testing parts
# training part (with prices) goes through the end of 2015, through record ID 98817
train = all_data[1:98818]
test = all_data[98818:]

# drop outliers: Price > 6000000
train.drop(train[train['Price'] > 6000000].index, inplace=True)

# We take the log here because the error metric is between the log of the
# SalePrice and the log of the predicted price. That does mean we need to
# exp() the prediction to get an actual sale price.
label_df = pd.DataFrame(index=train.index, columns=['Price'])
label_df['Price'] = np.log(train['Price'])

# drop ID col from train and test, save test ID as an object separately
testID = test['ID']
test.drop('ID', axis=1, inplace=True)
train.drop('ID', axis=1, inplace=True)

# drop Price col from testing and training
test.drop('Price', axis=1, inplace=True)
train.drop('Price', axis=1, inplace=True)

# Pre-processing and feature engineering tactics
# Price (training set) -> log transform
# Date -> YearOfSale, MonthOfSale -> Label Encoding -- or totally exclude?
# Old_New, Duration, PPD_Category_Type, Property_Type -> one-hot encoding
# Postcode, Street, Locality, Town, District, County -> calculate geolocation lat, long
# Town, District, County -> label encoding (factorize)
# Postcode, Street, Locality -> stripe out of the data set


print("-------------------------")
print("Preparing to fit regressor ensemble ... ")
print(dt.datetime.now() - start_time)


print('Training set size:', train.shape)
print('Test set size:', test.shape)

################################################################################
regr1 = xgb.XGBRegressor(
    colsample_bytree=0.2,
    gamma=0.0,
    learning_rate=0.005, # 0.01,
    max_depth=4,
    min_child_weight=1.5, # 1.5,
    n_estimators=30000,
    reg_alpha=0.9,
    reg_lambda=0.6,
    subsample=0.2,
    seed=42,
    silent=True)

best_alpha = 0.00098
regr2 = Lasso(alpha=best_alpha, max_iter=50000)

regr3 = ElasticNet(alpha=0.001)

regr4 = KernelRidge(alpha=0.3, kernel='polynomial', degree=2, coef0=1.85)

# Gradient Boosting Regression:
# With 'huber' loss that makes it robust to outliers
regr5 = ens.GradientBoostingRegressor(n_estimators=4000, learning_rate=0.005,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=15,
                                   loss='huber', random_state =5)

#regr = CustomEnsembleRegressor([regr1, regr2, regr3, regr4, regr5])
regr = CustomEnsembleRegressor([regr2, regr3])

# Evaluation was commented to make it run as  kernel
print('Evaluating each model separately..')
evaludate_submodels(regr, train, label_df)

# print('Evaluating ensemble..')
# evaludate_model(regr, train, label_df)

print('Fitting ensemble and predicting...')
# Fit the ensemble
regr.fit(train, label_df)

# Run prediction on the Kaggle test set.
y_pred = regr.predict(test)
print("Predictions competed in: ", dt.datetime.now() - start_time)

################################################################################
print('Saving results..')
# Blend the results of the ensemble regressor and save the prediction to a CSV file.
y_pred = np.exp(y_pred)

pred_df = pd.DataFrame(y_pred, index=testID, columns=['Price'])
pred_df.to_csv('output/ensemble_output_op.csv', header=True, index_label='ID')
print("Elapsed time: ", dt.datetime.now() - start_time)
print("Finished in: ", dt.datetime.now())