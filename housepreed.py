import numpy as np
import pandas as pd
import seaborn as sns
import math
import statsmodels.api as sm 
import keras
import pylab as py 
import matplotlib.pyplot as plt
train = pd.read_csv('trainData.csv')
test = pd.read_csv('testData.csv')
train=train.drop('Id',axis =1)  
test=test.drop('Id',axis =1)

A = train.drop("SalePrice",axis = 1,)
data=pd.concat((A,test)).reset_index()
data=data.drop('index',axis =1)  

for i in range(0,1460):
 train.loc[i,'SalePrice'] = math.log(train.loc[i,'SalePrice'])
Y = train.iloc[:,79]

for i in range(0,2919):
    if data.loc[i,'MSSubClass']==75:
       if (data.loc[i,'HouseStyle']=='2.5Fin'):
          data.loc[i,'MSSubClass']=73
       if (data.loc[i,'HouseStyle']=='2.5Unf'):
          data.loc[i,'MSSubClass']=77
#data[data['MSSubClass']==75]['HouseStyle']
data=data.drop('HouseStyle',axis=1)

data['years_remodel'] =data['YrSold']-data['YearRemodAdd']
data['years_built'] = data['YrSold']-data['YearBuilt']
data= data.drop(['YrSold','YearRemodAdd','YearBuilt','GarageYrBlt'],axis=1)

data[['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtFinType1','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','Fence','BsmtFinType2','BsmtExposure','MiscFeature','GarageCars','MiscVal']]=data[['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtFinType1','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','Fence','BsmtFinType2','BsmtExposure','MiscFeature','GarageCars','MiscVal']].fillna( 'None' )
data['Electrical']=data['Electrical'].fillna('SBrkr')
data['MasVnrArea']=data['MasVnrArea'].fillna(0)
data[['BsmtFullBath','BsmtHalfBath']]=data[['BsmtFullBath','BsmtHalfBath']].fillna(0)
data['SaleType']=data['SaleType'].fillna('WD')
data[['GarageArea','BsmtUnfSF','TotalBsmtSF','BsmtFinSF1']]=data[['GarageArea','BsmtUnfSF','TotalBsmtSF','BsmtFinSF1',]].fillna(0)
data['MSZoning']=data['MSZoning'].fillna('RM')
data['Exterior1st']=data['Exterior1st'].fillna('VinylSd') 
data['Exterior2nd']=data['Exterior2nd'].fillna('VinylSd')
data['Functional']=data['Functional'].fillna('Typ')
data['KitchenQual']=data['KitchenQual'].fillna('TA')

data.columns.to_series().groupby(data.dtypes).groups
spdata = ['MoSold','MSZoning', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood',
     'Condition1', 'BldgType', 'RoofStyle','Exterior1st', 'Exterior2nd', 'MasVnrType','Foundation',
     'BsmtFinType2',  'Electrical','Functional', 'GarageType','PavedDrive', 'Fence', 'MiscFeature',
     'SaleType','SaleCondition','MSSubClass','Street','Utilities','Condition2','RoofMatl','Heating',
     'PoolQC','GarageCars','MiscVal']

for i in range(0,1460):
    if train.loc[i,'MSSubClass']==75:
       if (train.loc[i,'HouseStyle']=='2.5Fin'):
          train.loc[i,'MSSubClass']=73
       if (train.loc[i,'HouseStyle']=='2.5Unf'):
          train.loc[i,'MSSubClass']=77
#data[data['MSSubClass']==75]['HouseStyle']
train=train.drop('HouseStyle',axis=1)

train['years_remodel'] = train['YrSold']-train['YearRemodAdd']
train['years_built'] = train['YrSold']-train['YearBuilt']
train = train.drop(['YrSold','YearRemodAdd','YearBuilt','GarageYrBlt'],axis=1)

train[['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtFinType1','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','Fence','BsmtFinType2','BsmtExposure','MiscFeature','GarageCars','MiscVal']]=train[['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtFinType1','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','Fence','BsmtFinType2','BsmtExposure','MiscFeature','GarageCars','MiscVal']].fillna( 'None' )
train['Electrical']=train['Electrical'].fillna('SBrkr')
train['MasVnrArea']=train['MasVnrArea'].fillna(0)
train[['BsmtFullBath','BsmtHalfBath']]=train[['BsmtFullBath','BsmtHalfBath']].fillna(0)
train['SaleType']=train['SaleType'].fillna('WD')
train[['BsmtUnfSF','TotalBsmtSF','BsmtFinSF1']]=train[['BsmtUnfSF','TotalBsmtSF','BsmtFinSF1',]].fillna(0)
train['MSZoning']=train['MSZoning'].fillna('RM')
train['Exterior1st']=train['Exterior1st'].fillna('VinylSd') 
train['Exterior2nd']=train['Exterior2nd'].fillna('VinylSd')
train['Functional']=train['Functional'].fillna('Typ')
train['KitchenQual']=train['KitchenQual'].fillna('TA')

train = train.drop(['TotRmsAbvGrd','1stFlrSF'], axis = 1) 
train = train.drop(['GarageArea'], axis = 1)
data = data.drop(['TotRmsAbvGrd','1stFlrSF'], axis = 1) 
data = data.drop(['GarageArea'], axis = 1)
cortrain = train.corr()
Saletrain=cortrain['SalePrice'] 
#plt.figure(figsize = (16,7)) 
#sns.heatmap(data.corr(), cmap='coolwarm',annot=False,linewidth =0.5)
#sns.heatmap([Sale], cmap='coolwarm',annot=False,linewidth =0.5)
train_features_1=cortrain[cortrain <=-0.8]
#train_features_2=cor[cor['SalePrice']<=-0.5]['SalePrice']
#corrr_features_1=cor[cor['SalePrice']>=0.5]['SalePrice']

data= pd.get_dummies(data=data,columns = spdata, drop_first = True)

function_map = {}
function_map['Ex'] = 5 #'Excellent'
function_map['Gd'] = 4 #'Good'
function_map['TA'] = 3 #'Average/Typical'
function_map['Fa'] = 2 #'Fair'
function_map['Po'] = 1 #'Poor'
function_map['NA'] = 0 #'NA'

data.ExterQual = data.ExterQual.map(function_map) 
data.GarageQual = data.GarageQual.map(function_map) 
data.GarageCond = data.GarageCond.map(function_map) 
data.FireplaceQu =data.FireplaceQu.map(function_map) 
data.KitchenQual = data.KitchenQual.map(function_map) 
data.HeatingQC = data.HeatingQC.map(function_map) 
data.BsmtCond = data.BsmtCond.map(function_map) 
data.BsmtQual = data.BsmtQual.map(function_map) 
data.ExterCond = data.ExterCond.map(function_map)

for_Garage ={}
for_Garage['Fin']=3
for_Garage['Rfn']=2
for_Garage['Unf']=1
for_Garage['None']=0

data.GarageFinish = data.GarageFinish.map(for_Garage) 

yesno={}
yesno['yes']=1
yesno['no']=0

data.CentralAir = data.CentralAir.map(yesno)

bsmt ={}
bsmt['Gd']=3
bsmt['Av']=2
bsmt['Mn']=1
bsmt['No']=0
bsmt['None']=0

data.BsmtExposure = data.BsmtExposure.map(bsmt)          

slope ={}
slope['Gtl'] =3
slope['Mod'] = 2
slope['Sev'] =1

data.LandSlope = data.LandSlope.map(slope)

bsmtfin = {}
bsmtfin['GLQ']=6
bsmtfin['ALQ']=5
bsmtfin['BLQ']=4
bsmtfin['Rec']=3
bsmtfin['LwQ']=2
bsmtfin['Unf']=1
bsmtfin['None']=0

data.BsmtFinType1 = data.BsmtFinType1.map(bsmtfin)
data=data.fillna(0)

from collections import Counter
def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers

test.columns.to_series().groupby(data.dtypes).groups

train_2 = pd.read_csv('trainData.csv')
B = train_2.drop("SalePrice",axis = 1,)
data_2=pd.concat((B,test)).reset_index()
for i in range(1460,2919):
    data_2['index'].loc[i] = data_2['index'].loc[i] +1460
data_2=data_2.drop('Id',axis =1)  

testFront = pd.concat((data_2['LotArea'],data_2['GrLivArea'],data_2['GarageCars']),axis = 1)
trainFront = pd.concat((data_2['LotArea'],data_2['GrLivArea'],data_2['GarageCars'],data_2['LotFrontage']),axis = 1)
trainFront = trainFront.fillna(0)
data_2 = data_2.fillna(0)
index_front = data_2['index']
for i in range(0,2919):
    if(trainFront.loc[i,'LotFrontage'] == 0):
        trainFront = trainFront.drop(index = i,axis = 0)
    else:
        testFront = testFront.drop(index = i,axis = 0)

for i in range(0,2919):
    if(data_2.loc[i,'LotFrontage'] != 0):
        index_front = index_front.drop(index = i,axis = 0)
    else:
        continue

trainFront_X = trainFront.iloc[:,:-1]
trainFront_y = trainFront.iloc[:,-1]
dataFront = pd.concat((trainFront_X,testFront), axis = 0)
dataFront = pd.get_dummies(data=dataFront ,columns = ['GarageCars'] , drop_first = True)
trainFront_X = dataFront.iloc[0:2433,:]
testFront = dataFront.iloc[2433:2919]
#cor_3 = trainFront.corr()
#Saletrain_3=cor_3['LotFrontage']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
col_name = ['LotArea', 'GrLivArea']
scaled_features = trainFront_X.copy()
features = scaled_features[col_name]
features = scaler.fit_transform(features)
scaled_features[col_name] = features 
trainFront_X = scaled_features.copy()

scaled_features = testFront.copy()
features_2 = scaled_features[col_name]
scaled_features[col_name] = scaler.transform(features_2)
testFront = scaled_features.copy() 

"""
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(scaled_features, trainFront_y)
"""

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
regressor = Lasso()
parameters = {'alpha' : [1e-15, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1, 3, 5, 10, 20, 30, 100]}
lasso_regressor = GridSearchCV(regressor, parameters, scoring = 'neg_mean_squared_error', cv = 5)

lasso_regressor.fit(trainFront_X, trainFront_y)
testFront_y = lasso_regressor.predict(testFront)

data.loc[index_front,'LotFrontage']= testFront_y

data['LotDepth'] = data['LotArea']/data['LotFrontage']

X = data.loc[0:1459,:]
testData = data.loc[1460:2919,:]

train = pd.concat((X,Y),axis = 1)

ind = detect_outliers(X, 1, ['LotArea','GrLivArea'])
train = train.drop(ind,axis = 0)
ind_2 = detect_outliers(train, 1, ['SalePrice'])

X = train.iloc[:,:-1]
Y = train.iloc[:,-1]

col_name = ['LotFrontage','LotArea', 'MasVnrArea','BsmtFinSF1', 'BsmtFinSF2',
            'BsmtUnfSF', 'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'LowQualFinSF',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'
            , 'PoolArea', 'years_remodel', 'years_built']
scaled_features_2 = X.copy()
features_3 = scaled_features_2[col_name]
features_3 = scaler.fit_transform(features_3)
scaled_features_2[col_name] = features_3 
X = scaled_features_2.copy()

scaled_features_4 = testData.copy()
features_4 = scaled_features_4[col_name]
scaled_features_4[col_name] = scaler.transform(features_4)
testData = scaled_features_4.copy() 

regressor_2 = Lasso()
lasso_regressor_2 = GridSearchCV(regressor_2, parameters, scoring = 'neg_mean_squared_error', cv = 5)

lasso_regressor_2.fit(X,Y)
predict_y = lasso_regressor_2.predict(testData)

e=math.exp(1)
Y_final= e**predict_y

again = pd.read_csv('testData.csv')

cortrain = train.corr()
Saletrain=cortrain['SalePrice']

output = pd.DataFrame({'Id': again.Id, 'SalePrice': Y_final})
output.to_csv('my_submission_2.csv', index=False)
