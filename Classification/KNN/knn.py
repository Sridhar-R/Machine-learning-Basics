import pandas as pd
import matplotlib
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split

names = ['sepal_length','sepal_width','petal_length','petal_width', 'label']

df = pd.read_csv('iris.data', header = None, names= names)
label_dict = {'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
X = np.array(df.ix[:,0:4])
y = np.array(df['label'])
print (len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
print (len(X_train))
#plotting_data
colors = ['red','green','blue']
fig = plt.figure(figsize = (8,8))
plt.scatter(df["petal_length"],df["petal_width"],c=colors, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

def train(x_train, y_train):
    #do_nothing
    return
def predict(x_train, y_train, x_test, k):
    distances = []
    targets = []
    for i in range(len(x_train)):
        distance = np.sqrt(np.sum(np.square(x_test - x_train[i,:])))
        distances.append([distance, i])
    distances = sorted(distances)
    #print (distances)
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])
    return Counter(targets).most_common(1)[0][0]

def knn(x_train, y_train, x_test, predictions, k):
    train(x_train,  y_train)
    for i in range(len(x_test)):
        val = predict(x_train, y_train, x_test[i,:], k)
        predictions.append(val)
predictions = []

knn(X_train, y_train, X_test, predictions, 60)
predictions = np.array(predictions)
accuracy = accuracy_score(y_test, predictions)
print('\nThe accuracy of our classifier is', (accuracy*100))

print ("Testing Conditions")
new_data = []
for i in range(4):
    temp_ = float(input(("Enter the" ,i,"th input")))
    new_data.append(temp_)
new_predict = predict(X_train,y_train, new_data, 7)
print (new_predict)
