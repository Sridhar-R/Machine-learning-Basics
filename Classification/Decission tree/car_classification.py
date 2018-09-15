import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
dataread=pd.read_csv("car_data.txt",header=None, names=["buying","maintain","person","doors","lug_boot","safety","class"])
dataread.head()
#print (dataread)


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
	('encoding',MultiColumnLabelEncoder(columns=['buying','maintain','lug_boot','safety','class']))
])
dataread = encoding_pipeline.fit_transform(dataread)
out = dataread.ix[:,6:7]
#out=np.array(out)
inp = dataread.ix[:,0:6]
#inp=np.array(inp)
inp.columns.tolist()
inp['person']=inp['person'].replace(['5more'] , 5)
inp['doors']=inp['doors'].replace(['more'], 5)
print(inp)
from sklearn.cross_validation import train_test_split
train_inp,test_inp,train_out,test_out=train_test_split(inp,out,train_size=0.66,test_size=0.33)
print(np.shape(train_inp))
print (dataread.head())
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=4, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
model.fit(train_inp, train_out)
print(model)
train_pred=model.predict(train_inp)
test_pred=model.predict(test_inp)
print("Train Prediction = ",train_pred)
print(np.shape(test_pred))
print("Test Prediction = ", test_pred)
print(np.shape(test_pred))
from sklearn.metrics import accuracy_score
train_acc=accuracy_score(train_out,train_pred)
print("Train Accuracy = ",train_acc)
test_acc = accuracy_score(test_out,test_pred)
print("Test Accuracy =",test_acc)