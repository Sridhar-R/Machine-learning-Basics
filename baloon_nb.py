import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
dataread = pd.read_csv("baloon_data.txt",header=None, names=["color","size","act","age","inflated"])
dataread=dataread.apply(LabelEncoder().fit_transform)
#print (dataread)
oup = dataread.ix[:,4:5]
inp = dataread.ix[:,0:4]
print(dataread.head())

from sklearn.cross_validation import train_test_split
inp_train, inp_test, oup_train, oup_test = train_test_split(inp, oup, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
inp_train = sc.fit_transform(inp_train)
inp_test = sc.transform(inp_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(inp_train, oup_train)

oup_pred = classifier.predict(inp_test)
print(oup_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(oup_test, oup_pred)
print(cm)

from matplotlib.colors import ListedColormap
inp_set, oup_set = inp_train, oup_train

inp1, inp2 = np.meshgrid(np.arange(start = inp_set[:, 0].min() - 1, stop = inp_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = inp_set[:, 1].min() - 1, stop = inp_set[:, 1].max() + 1, step = 0.01))

plt.contourf(inp1, inp2, classifier.predict(np.array([inp1.ravel(), inp2.ravel()]).T).reshape(inp1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.contourf(inp1, inp2, classifier.predict(np.array([inp1.ravel(), inp2.ravel()]).T).reshape(inp1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(inp1.min(), inp1.max())
plt.ylim(inp2.min(), inp2.max())
for i, j in enumerate(np.unique(oup_set)):
    plt.scatter(inp_set[oup_set == j, 0], inp_set[oup_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Naive Bayes (Training set)')
plt.xlabel('Color')
plt.ylabel('Inflated')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
inp_set, oup_set = inp_test, oup_test
inp1, inp2 = np.meshgrid(np.arange(start = inp_set[:, 0].min() - 1, stop = inp_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = inp_set[:, 1].min() - 1, stop = inp_set[:, 1].max() + 1, step = 0.01))
plt.contourf(inp1, inp2, classifier.predict(np.array([inp1.ravel(), inp2.ravel()]).T).reshape(inp1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(inp1.min(), inp1.max())
plt.ylim(inp2.min(), inp2.max())
for i, j in enumerate(np.unique(oup_set)):
    plt.scatter(inp_set[oup_set == j, 0], inp_set[npoup_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Color')
plt.ylabel('Inflated')
plt.legend()
plt.show()
