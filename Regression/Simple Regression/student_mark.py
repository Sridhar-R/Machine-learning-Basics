import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataread=pd.read_csv("students_score.CSV")
inp=dataread.iloc[:,:-1].values
oup=dataread.iloc[:,1].values
print(dataread.head())
from sklearn.cross_validation import train_test_split
train_inp,test_inp,train_oup,test_oup=train_test_split(inp,oup,test_size=1/3, random_state=1)

"""from sklearn.preprocessing import StandardScaler
sc_inp=StandardScaler()
train_inp=sc_inp.fit_transform(train_inp)
test_inp=sc_inp.fit_transform(test_inp)
sc_oup=StandardScaler()
train_oup=sc_oup.fit_transform(train_oup)"""

from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(train_inp,train_inp)

pred_oup = regressor.predict(test_oup)
#print(pred_oup)

plt.scatter(train_inp,train_oup,color='red')
plt.plot(train_inp,regressor.predict(train_inp),color = 'blue')
plt.title('Hours Vs Score (Training set)')
plt.xlabel('Student Score')
plt.ylabel('Time Spent')
plt.show()

plt.scatter(test_inp,test_oup,color='red')
plt.plot(test_inp,regressor.predict(test_inp),color = 'blue')
plt.title('Hours Vs Score (Test set)')
plt.xlabel('Student Score')
plt.ylabel('Time Spent')
plt.show()
