#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import seaborn as sns


#bring in the six packs
df_train = pd.read_csv('data.csv')

#check the decoration
#print(df_train.columns)


#descriptive statistics summary
#print(df_train['Price'].describe())

#skewness and kurtosis
print("Skewness: %f" % df_train['Price'].skew())
print("Kurtosis: %f" % df_train['Price'].kurt())

#print(df_train.dtypes)


#correlation matrix
#corrmat = df_train.corr()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True);

sp_corr = df_train.corr()["Price"]
sp_corr_sort = sp_corr.sort_values(axis = 0 , ascending = False)
#print(sp_corr_sort[sp_corr_sort > 0])

#missing data
#total = df_train.isnull().sum().sort_values(ascending=False)
#percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(20))

#Skewed variables affect the performance of a Regression ML model so we do a Log transform to remove skewness
from scipy.stats import skew
#log transform the target:
df_train["Price"] = np.log1p(df_train["Price"])
#print(df_train.head())

#log transform skewed numeric features:
numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index

skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

df_train[skewed_feats] = np.log1p(df_train[skewed_feats])

#print(df_train.head())

# We take the log here because the error metric is between the log of the
# SalePrice and the log of the predicted price. That does mean we need to
# exp() the prediction to get an actual sale price.
#label_df = pd.DataFrame(index = df_train.index, columns=["Price"])
#label_df["Price"] = np.log(df_train["Price"])

#print("Training set size:", label_df.shape)

categorical = df_train.select_dtypes(exclude=['float64', 'int64'])


#print(df_train.select_dtypes(include=['object']).columns)

#Numerical Columns
#print(df_train.select_dtypes(include=['float64', 'int64']).columns)

df_train['Postcode'].fillna('None',inplace = True)

df_train['Locality'].fillna('None',inplace = True)

#Missing values in the columns
print(df_train[df_train.columns[df_train.isnull().any()]].isnull().sum())

from sklearn.preprocessing import LabelEncoder

labelEnc=LabelEncoder()


cat_vars = ['Date', 'Postcode', 'Property_Type', 'Old_New', 'Duration', 'Street', 'Locality', 'Town', 'District', 'County','PPD_Category_Type',]

for col in cat_vars:
    df_train[col]=labelEnc.fit_transform(df_train[col])
print(df_train.head())

New_Train = df_train[:70000]
#print(New_Train)
X_train = New_Train.drop('Price',axis=1)
y_train = New_Train['Price']

print(New_Train.shape)


New_Test = df_train[70000:]
#print(New_Test)
X_test = New_Test.drop('Price',axis=1)

print(X_test.shape)


from sklearn.linear_model import Ridge,RidgeCV
from sklearn.model_selection import cross_val_score

#Defining a function to calculate the RMSE for each Cross validated fold
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return (rmse)

#model_ridge = Ridge(alpha = 5).fit(X_train, y_train)

#Finding the optimum Alpha value using cross validation
alphas = [0.0001,0.1,0.5,1,2,5,7,10]
rmse_cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
print(rmse_cv_ridge)

model_ridge = Ridge(alpha = 2).fit(X_train, y_train)

print(rmse_cv(model_ridge).mean())

ridge_preds = np.exp(model_ridge.predict(X_test))

print("Ridge prediction ::")
print(0.5*ridge_preds)

from sklearn.linear_model import Lasso

#model_lasso = Lasso().fit(X_train, y_train)

#alphas = [0.00001, 0.0001, 0.001,0.002, 0.005, 0.01]
#rmse_cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]
#print(rmse_cv_lasso)


model_lasso = Lasso(alpha = 0.001 , max_iter=5000).fit(X_train, y_train)

print(rmse_cv(model_lasso).mean())

lasso_preds = np.expm1(model_lasso.predict(X_test))
print("Lasso prediction ::")
print(0.5*lasso_preds)

Final = 0.5*lasso_preds + 0.5*ridge_preds # combining the models

submission = pd.DataFrame({"ID": X_test["ID"],"Price": Final})

submission.to_csv("HousePrice.csv", index=False)






