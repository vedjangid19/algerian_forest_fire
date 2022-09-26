import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

pd. set_option('display.max_columns', 500)
pd. set_option('display.width', 1000)

def show_hist(data):
    plt.rcParams["figure.figsize"] = 15,18
    data.hist()
    plt.show()

def show_PairPlot(data):
    sns.pairplot(data)
    # plt.show()

def outliers(x):
    q1,q3 = np.percentile(x,[25,75])
    iqr = q3-q1
    lower_fence = q1-1.5*(iqr)
    highr_fence = q3+1.5*(iqr)
    print("Q1, Q3, IQR, Lower_Fence, Higher_Fence : ",q1,q3,iqr,lower_fence,highr_fence)
    ourlier = (x.loc[(x < lower_fence) | (x > highr_fence)])
    return ourlier

dataset = pd.read_csv("../hour.csv")
print(dataset.head(2))
print("Dataset Shape : ",dataset.shape)

# show_hist(dataset)
# show_PairPlot(dataset)

# missing values
missing_value = dataset.isnull().sum()
# print("missing Values : ",missing_value)

dataset['Year'] = dataset.dteday.str.split("-").str[0]
dataset['Year'] = dataset.Year.astype(int)

dataset['Month'] = dataset.dteday.str.split("-").str[1]
dataset['Month'] = dataset.Month.astype(int)

dataset['Date'] = dataset.dteday.str.split("-").str[1]
dataset['Date'] = dataset.Date.astype(int)

dataset.drop(['dteday','yr','mnth'], axis=1, inplace=True)
print(dataset.head(2))

cnt_outliers = outliers(dataset['cnt']).count()
print("cnt outliers : ",cnt_outliers)

# REMOVE OUTLIERS
q1,q3 = np.percentile(dataset.cnt,[25,75])
iqr = q3-q1
lower_fence = q1-1.5*(iqr)
highr_fence = q3+1.5*(iqr)
print("OLD Dataset Shape : ",dataset.shape)

new_dataset = dataset[(lower_fence < dataset['cnt']) & (dataset['cnt'] < highr_fence)]
print("New Dataset Shape : ",new_dataset.shape)


#STATISTICS TEST
from scipy.stats import anderson
print("ANDERSON TEST : ",anderson(new_dataset['cnt']))

from scipy.stats import shapiro
print("SHAPIRO TEST : ",shapiro(new_dataset['cnt']))

from scipy.stats import kstest
print("KSTEST : ",kstest(new_dataset['cnt'],"norm"))


from scipy import stats
from sklearn.preprocessing import PowerTransformer
print("DataFRame columns : ",dataset.columns)
pt = PowerTransformer(method='yeo-johnson', standardize=True)
df = pt.fit_transform(dataset)
df = pt.fit_transform(new_dataset)

# print("Fit Transform : ",df)
df = pd.DataFrame(df)

print(df.head(2))

df. rename(columns= {0:'instant',1:'season',2:'hr',3:"holiday",4:"weekday",5:"workingday",6:"weathersit",7:"temp",8:"atemp",9:"hum",10:"windspeed",11:"casual",12:"registered",13:"cnt",14:"Year",15:"Month",16:"Date"},inplace=True)

print(df.head(2))

# show_hist(df)

X = df.drop(['cnt'], axis=1)
y = df['cnt']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=40)


# import statsmodels.api as sm
# model2 = sm.OLS(y_train,X_train).fit()
# model_summary = model2.summary()
# print("Model Summary : ",model_summary)


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=0, normalize=True)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(regressor, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
print("Scores : ",scores)
print("MSE : ",np.mean(scores))

from sklearn.metrics import r2_score

Linear_regressor_train = regressor.fit(X_train,y_train)
Linear_regressor_test_pred = regressor.predict(X_test)

from sklearn import metrics
print('ROOT EAN ERROR SQUARE:',np.sqrt(metrics.mean_squared_error(y_test,Linear_regressor_test_pred)))


score = r2_score(y_test,Linear_regressor_test_pred)
score1 = regressor.score(X_train,y_train)
print("Score : ",score)
print("Score1 : ",score1)


import pickle
pickle.dump(regressor,open('model.pkl', 'wb'))

# pickle.load('model.pkl', 'rb')


