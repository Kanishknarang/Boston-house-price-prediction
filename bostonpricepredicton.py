#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

#loadind data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#shape
print('Train data shape:',train.shape)
print('Test data shape:',test.shape)

#printing first few lines
print(train.head())
print(test.head())

#initializing plot
plt.style.use(style="ggplot")
plt.rcParams['figure.figsize']=(10,6)

#SalePrice description
print(train.SalePrice.describe())

#Skewness of SalePrice
print(train.SalePrice.skew())
plt.hist(train.SalePrice,color="blue")
plt.show()

#taking log for target values to achive normal distribution
target = np.log(train.SalePrice)
print(target.skew())
plt.hist(target,color='blue')
plt.show()

#finding correlation between numeric features
numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5])
print(corr['SalePrice'].sort_values(ascending=False)[-5:])


#scatter plot 
plt.scatter(x=train['GarageArea'],y=target)
plt.xlabel = 'GarageArea'
plt.ylabel = 'SalePrice'
plt.show()


#removing outliers
train = train[train['GarageArea']<1200]

plt.scatter(x=train['GarageArea'],y=np.log(train['SalePrice']))
plt.xlim(-200,1600)
plt.xlabel = 'GarageArea'
plt.ylabel = 'SalePrice'
plt.show()

nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending = False)[:25])
nulls.columns = ['Null count']
nulls.index.name = 'feature'
print(nulls)

categoricals = train.select_dtypes(exclude = [np.number])
print(categoricals.describe())

print(train.Street.value_counts())

train['enc_street'] = pd.get_dummies(train.Street,drop_first = True)
test['enc_street'] = pd.get_dummies(test.Street,drop_first = True)

print(train.enc_street.value_counts())

condition_pivot = train.pivot_table(index = 'SaleCondition',values = 'SalePrice',aggfunc = np.median)
condition_pivot.plot(kind= 'bar',color='blue')

plt.xticks(rotation=0)
plt.show()

def encode(x):
    if x=='Partial':
        return 1
    else:
        return 0

train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

condition_pivot = train.pivot_table(index = 'enc_condition',values = 'SalePrice',aggfunc = np.median)
condition_pivot.plot(kind= 'bar',color='blue')

plt.xticks(rotation=0)
plt.show()

data = train.select_dtypes(include= [np.number]).interpolate().dropna()
print(sum(data.isnull().sum()!=0))

y = np.log(train.SalePrice)
X= data.drop(['SalePrice','Id'],axis=1)
print(y)
print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,y, random_state=42,test_size=.33)

lr = linear_model.LinearRegression()

model = lr.fit(X_train,Y_train)

print(model.score(X_test,Y_test))

predictions = model.predict(X_test)

print(mean_squared_error(Y_test,predictions))

submission = pd.DataFrame()
submission['Id'] = test.Id

feats = test.select_dtypes(include=[np.number]).drop(['Id'],axis=1).interpolate()

predictions = model.predict(feats)

final_predictions = np.exp(predictions)

submission['SalePrice'] = final_predictions
print(submission.head())
print(test)

submission.to_csv('submission1.csv',index=False)
