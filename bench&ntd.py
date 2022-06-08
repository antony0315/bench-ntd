import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

# %matplotlib inline
df=pd.read_csv('C:/Users/user/Desktop/learn_python/匯率與加權指數/benchNTD.csv',usecols=[1,2],header=1)
df=df.dropna()


bench=np.array(df.iloc[:,0]).reshape(-1, 1)
ntd=np.array(df.iloc[:,1]).reshape(-1, 1)
x_test=pd.DataFrame([27.5,31.5])

reg_model=linear_model.LinearRegression()
reg_model.fit(ntd,bench)

y_test_predict=reg_model.predict(x_test)
print(y_test_predict)

plt.figure()
plt.scatter(ntd,bench)
plt.plot([27.5,31.5],[17382.155,8760.4885],color='red')

print('coefficients:',reg_model.coef_)

from sklearn.metrics import r2_score
print(r2_score(bench,reg_model.predict(ntd)))
