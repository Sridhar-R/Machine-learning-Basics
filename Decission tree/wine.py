import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
dataread=pd.read_csv("wine_data.txt",header=None, names=["class","Alcohol",",Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"])
dataread.head()
print (dataread)
class MultiColumnLabelEncoder:
	def __init__(self,columns = None):
		self.columns = columns
	def fit(self,x,y=None):
		return self
	def transform(self,x):
		output = x.copy()
		if self.columns is not None:
			for col in self.columns:
				output [col] =LabelEncoder().fit_transform(output[col])
		else:
			for colname,col in output.iteritems():
				output[colname] =LabelEncoder().fit_transform(col)
		return output
	def fit_transform(self,x,y=None):
		return self.fit(x,y).transform(x)
encoding_pipeline = Pipeline([
	("encoding",MultiColumnLabelEncoder(columns=['class','Alcohol',',Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']))
])
dataread = encoding_pipeline.fit_transform(dataread)
out = dataread.ix[:,0:1]
inp = dataread.ix[:,1:14]
inp.columns.tolist()
#print(inp)
#print (out)
from sklearn.cross_validation import train_test_split
train_inp1,test_inp1,train_out1,test_out1=train_test_split(inp,out,train_size=0.75,test_size=0.25)
train_inp2,test_inp2,train_out2,test_out2=train_test_split(inp,out,train_size=0.75,test_size=0.25)
print(np.shape(train_inp1))
print(np.shape(train_inp2))
print (dataread.head())

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(train_inp1, train_out1)
print(model)
train_dt_pred=model.predict(train_inp1)
test_dt_pred=model.predict(test_inp1)
print ("Train Prediction of Decisiontree = " ,train_dt_pred)
print (np.shape(train_dt_pred))
print("Test Pediction of Decisiontree= " ,test_dt_pred)
print(np.shape(test_dt_pred))
from sklearn.metrics import accuracy_score
train_dt_acc = accuracy_score(train_out1,train_dt_pred)
print("Train Accuracy of DecisionTree = ",train_dt_acc)
test_dt_acc=accuracy_score(test_out1, test_dt_pred)
print("Test Accuracy of DecisionTree = ",test_dt_acc)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
model.fit(train_inp2, train_out2)
print(model)
train_rf_pred=model.predict(train_inp2)
test_rf_pred=model.predict(test_inp2)
print("Train Prediction of Randomforest = ",train_rf_pred)
print(np.shape(test_rf_pred))
print("Test Prediction of Randomforest= ", test_rf_pred)
print(np.shape(test_rf_pred))
from sklearn.metrics import accuracy_score
train_rf_acc=accuracy_score(train_out2,train_rf_pred)
print("Train Accuracy of Randomforest = ",train_rf_acc)
test_rf_acc = accuracy_score(test_out2,test_rf_pred)
print("Test Accuracy of Random forest =",test_rf_acc)
