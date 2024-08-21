import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv",header=1)


#datacleaning

#misisng values

missing_values_df = df[df.isnull().any(axis=1)]
print(missing_values_df)

df.loc[:122,'region'] = 0
df.loc[122:,'region'] = 1
df[['region']] = df[['region']].astype(int)
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

#remove 122 becuz it contains headings
df.drop(index=122,inplace=True)
df.reset_index(drop=True,inplace=True)
print(df.head())

#remove spaces in column names

df.columns = df.columns.str.strip()


#change the required columns as intDatatype and floatdatatype
df[['month','day','year','Temperature','RH','Ws']] = df[['month','day','year','Temperature','RH','Ws']].astype(int)

features = []

for feature in df.columns:
    if df[feature].dtype == 'object':
        features.append(feature)

for i in features:
    if i!='Classes':
        df[i] = df[i].astype('float')

df.info()
#df.to_csv("cleaned_Dataset.csv",index=False)
df_copy = df.copy()
df_copy = df_copy.drop(['day','month','year'],axis=1)

#encoding caetgories as it is words into 0s and 1s

df_copy['Classes'] = np.where(df_copy["Classes"].str.contains('not fire'),0,1)


print(df_copy["Classes"].value_counts())

#independatn amd dependant features

x = df_copy.drop("FWI",axis=1)
y= df_copy["FWI"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

#feature selection
print(x_train.corr())

#multicolinearity = independant features should not be correlated a lot

def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

corr_ftrs = correlation(x_train,0.90)#90 >correlated return the colimns

x_train.drop(corr_ftrs,axis=1,inplace=True)
x_test.drop(corr_ftrs,axis=1,inplace=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled= scaler.transform(x_test)

#linearregression

from sklearn.linear_model import LinearRegression

lin = LinearRegression()
lin.fit(x_train_scaled,y_train)
y_pred = lin.predict(x_test_scaled)

#lasso regression:
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(x_train_scaled,y_train)
y_pred = lasso.predict(x_test_scaled)

#ridge 

from sklearn.linear_model import Ridge
rid = Ridge()
rid.fit(x_train_scaled,y_train)
y_pred = rid.predict(x_test_scaled)

#elastic net
from sklearn.linear_model import ElasticNet
en = ElasticNet()
en.fit(x_train_scaled,y_train)
y_pred = en.predict(x_test_scaled)

#cv with lasso __HYPER PARAMETER TUNING ACCURACY 91 TO 98

from sklearn.linear_model import LassoCV
lsv = LassoCV(cv=5)
lsv.fit(x_train_scaled,y_train)
y_pred = lsv.predict(x_test_scaled)
print(lsv.alpha_)
from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)
print("r2",score)

new_data = np.array([[29,57,18,0,65.7,3.4,7.6,1.3,3.4,0]])
new_data_scaled = scaler.transform(new_data)
final1 = lsv.predict(new_data_scaled)
print(final1)

import pickle
pickle.dump(scaler,open('scaler.pkl','wb'))
pickle.dump(rid,open('ridge.pkl','wb'))


