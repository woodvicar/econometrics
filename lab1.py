import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df =pd.read_csv(r"/Users/vicar/Library/Mobile Documents/com~apple~CloudDocs/Dokument – Victoria’s MacBook Air/GU/Master /Financial Econometrics/lab1_data_F74EFA/data/data1.csv", header = [0])
print(df)
print(df.describe())

corrmatrix = df.corr()

print(corrmatrix)


f = plt.figure()
ax = plt.subplot(111)
df.plot.scatter('x1', 'y', ax=ax)
plt.show()

ax2 = plt.subplot(122)
df.plot.scatter('x2', 'y', ax=ax2)
plt.show()

f = plt.figure(figsize=(10,10))
labels=['y','x1','x2','w1']
ii=0
for i,l in enumerate(labels):
    for i2,l2 in enumerate(labels):
        ii+=1
        if i2>i:
            continue
        elif i2==i:
            df[[l]].plot.hist(ax=plt.subplot(4,4,ii))
        else:
            df.plot.scatter(x=l, y=l2, ax=plt.subplot(4,4,ii))
        
plt.show()

##OLS 

data = pd.read_csv("/Users/vicar/Library/Mobile Documents/com~apple~CloudDocs/Dokument – Victoria’s MacBook Air/GU/Master /Financial Econometrics/lab1_data_F74EFA/data/data1.csv", usecols=['y']) # regressed variable
data1 = pd.read_csv("/Users/vicar/Library/Mobile Documents/com~apple~CloudDocs/Dokument – Victoria’s MacBook Air/GU/Master /Financial Econometrics/lab1_data_F74EFA/data/data1.csv", usecols=['x1','x2']) # regressors
x = data
y = data1
data = np.squeeze(np.array(data))
data1 = np.squeeze(np.array(data1))

x = data
y = data1


A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m, c)
betahat, resid, rnk, singvals = np.linalg.lstsq(A, y)
print(betahat, resid, rnk, singvals)

##IV - no time and skills...


###dataset 2

df2 =pd.read_csv(r"/Users/vicar/Library/Mobile Documents/com~apple~CloudDocs/Dokument – Victoria’s MacBook Air/GU/Master /Financial Econometrics/lab1_data_F74EFA/data/data2.csv", header = [0], sep='\t')
print(df2.head(5))
print(df2.describe())

corrmatrix2 = df2.corr()

print(corrmatrix2)


f = plt.figure()
ax = plt.subplot(111)
df2.plot.scatter('x1', 'i   y', ax=ax)
plt.show()

f = plt.figure(figsize=(10,10))
labels=['i   y','x1','x2','w1']
ii=0
for i,l in enumerate(labels):
    for i2,l2 in enumerate(labels):
        ii+=1
        if i2>i:
            continue
        elif i2==i:
            df2[[l]].plot.hist(ax=plt.subplot(4,4,ii))
        else:
            df2.plot.scatter(x=l, y=l2, ax=plt.subplot(4,4,ii))
        
plt.show()



data2 = pd.read_csv("/Users/vicar/Library/Mobile Documents/com~apple~CloudDocs/Dokument – Victoria’s MacBook Air/GU/Master /Financial Econometrics/lab1_data_F74EFA/data/data2.csv", usecols=['i   y'], sep='\t') # regressed variable
data3 = pd.read_csv("/Users/vicar/Library/Mobile Documents/com~apple~CloudDocs/Dokument – Victoria’s MacBook Air/GU/Master /Financial Econometrics/lab1_data_F74EFA/data/data2.csv", usecols=['x1', 'x2'], sep='\t') # regressors
x = data2
y = data3
data2 = np.squeeze(np.array(data2))
data3 = np.squeeze(np.array(data3))

x = data2
y = data3


A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m, c)
betahat, resid, rnk, singvals = np.linalg.lstsq(A, y)
print(betahat, resid, rnk, singvals)

