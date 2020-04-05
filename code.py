import pandas as pd
import numpy as np
import matplotlib as plt
import glob
import csv



print(glob.glob('.../data/*.csv'))

#calculate daily realised variance
RVstatsq_1 = []

for filename in glob.glob('.../data/*.csv'):
    df = pd.read_csv(filename, parse_dates =["TIMESTAMP"], index_col ="TIMESTAMP")
    df.resample('30T', label='right', closed='right').last()
    df['log_price'] = np.log(df['PRICE'])
    df['daily_returns'] = (df['log_price']- df['log_price'].shift(1))
    df.dropna(inplace = True)
    df['returnssq'] = (df['daily_returns']**2)
    RVstatsq = df['returnssq'].sum()
    RVstatsq_1.append(RVstatsq)

pd.DataFrame(RVstatsq_1).to_csv("RVstatsq.csv")

#plot the results

df1 = pd.read_csv('.../RVstatsq.csv', header = None)
df1[:10]
df1.plot()


#calculate daily realised volatility 

RVstat_2 = []
for filename in glob.glob('.../data/*.csv'):
    df = pd.read_csv(filename, parse_dates =["TIMESTAMP"], index_col ="TIMESTAMP")
    df.resample('30T', label='right', closed='right').last()
    df['log_price'] = np.log(df['PRICE'])
    df['daily_returns'] = (df['log_price']- df['log_price'].shift(1))
    df.dropna(inplace = True)
    df['returnssq'] = (df['daily_returns']**2)
    RVstatsq = df['returnssq'].sum()
    RVstat = RVstatsq**0.5
    RVstat_2.append(RVstat)


print(RVstat_2)
pd.DataFrame(RVstat_2).to_csv("RVstat.csv")

df2 = pd.read_csv('.../RVstat.csv', header = None)
df2[:10]
df2.plot()

#calculate log daily realised variance 

logRVstat_1 = []

for filename in glob.glob('.../data/*.csv'):
    df = pd.read_csv(filename, parse_dates =["TIMESTAMP"], index_col ="TIMESTAMP")
    df.resample('30T', label='right', closed='right').sum()
    df['log_price'] = np.log(df['PRICE'])
    df['daily_returns'] = (df['log_price']- df['log_price'].shift(1))
    df.dropna(inplace = True)
    df['returnssq'] = (df['daily_returns']**2)
    RVstatsq = df['returnssq'].sum()
    RVstat = RVstatsq**0.5
    logRVstat = np.log(RVstat)
    print(logRVstat)
    logRVstat_1.append(logRVstat)

pd.DataFrame(logRVstat_1).to_csv('logRVstat.csv')   

df3 = pd.read_csv('.../logRVstat.csv', header = None)
df3[:10]
df3.plot()

#HAR-RV

#transform the dependent variables

df4 = pd.read_csv('.../RVstatsq.csv', header = None, names=['index', 'daily'])
df4[:10]


#weekly RV

df4['weekly'] = df4['daily'].rolling(window=5, min_periods=5).mean()
df4[:30]
df4.dropna(inplace = True)


#monthly RV
df4['monthly'] = df4['daily'].rolling(window=22, min_periods=22).mean()
df4[:30]
df4.dropna(inplace=False)


#OLS 

import statsmodels.api as sm

#forecasted values

df4['forecast']=df4['daily'].shift(-1)
df4.dropna()
df4.head(50)
print(df4)
df4.dropna(inplace=True)

#estimate coefficients
X = df4[['daily','monthly','weekly']] 
Y = df4['forecast']

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':1000})


model_details = model.summary()
print(model_details)
print(model.summary().as_latex())


#check the forecasting accuracy. I know this was not a part of the assignment but I got inspired. 
#There are certainly other ways to test for accuracy but I stumbled upon this one in the literature. 
#The code is running but I feel there is something incorrect about it. If there is time, I would like to find out what went wrong. 



#split dataframe in two parts
def split(df, headSize) :
    hd = df.head(headSize)
    tl = df.tail(len(df)-headSize)
    return hd, tl

first, second = split(df4, 400)

first.head(100)
second.head(100)

#run HAR on the first part

X = first[['daily','monthly','weekly']] 
Y = first['forecast']

X = sm.add_constant(X)

model2 = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':1000})


model2_details = model.summary()
print(model2_details)

#estimate a forecast with the obtained coefficients

nsample = 400
sig = 1
X = second[['daily','monthly','weekly']] 
X = sm.add_constant(X)
beta = [0.0569, 6.769e-05, 0.3751, 0.04770, ]
y_true = np.dot(X, beta)
Y = y_true + sig * np.random.normal(size=nsample)

olsmod = sm.OLS(Y, X)
olsres = olsmod.fit()
print(olsres.summary())

#assume standard normal error
test = []
nsample = 400
sig = 1
test = 6.769e-05 + 0.0569*first['daily'].shift(-1) +  0.3751*first['monthly'].shift(-1) + 0.04770*first['weekly'].shift(-1) + sig * np.random.normal(size=nsample)
test.head(100)

pd.DataFrame(test).to_csv('test.csv') 
test.head(100)

#compare

test['actual'] = pd.concat(first['daily'], axis=1)
test.head(400)

