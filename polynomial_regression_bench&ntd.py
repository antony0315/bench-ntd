import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
#%matplotlib inline
df=pd.read_csv('C:/Users/anton/OneDrive/桌面/learn_python/匯率與加權指數/benchNTD.csv',usecols=[1,2],header=1)
df=df.dropna()

x =pd.DataFrame(df.iloc[:,1].values)
y =pd.DataFrame(df.iloc[:,0].values)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree = 2)  #控制多项式的度
X_poly = poly.fit_transform(x)       #多變數
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)

plt.figure(dpi=1000)
plt.scatter(x, y, color = 'blue')
plt.plot(np.asarray(x), lin2.predict(X_poly), color = 'red')
plt.title('加權指數與台幣匯率關係')
plt.xlabel('NTD')
plt.ylabel('bench')
 
from sklearn.metrics import r2_score
print(r2_score(y,lin2.predict(poly.fit_transform(x))))

print('係數:',lin2.coef_) 
print('截距',lin2.intercept_)



from statsmodels.api import OLS
OLS(y,x).fit().summary()





