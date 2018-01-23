import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

#load the data using pandas
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

#replace the missing data represented by ? just a larger number
df.replace('?', -99999, inplace=True)

#delete the first column which shows IDs of patients as it will create a outlier
df.drop(['id'], 1, inplace=True)

#features everything except for class
X = np.array(df.drop(['class'], 1))

#labels only class
y = np.array(df['class'])

#shuffle the data and seprate the training and test data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

#example to predict never seen before
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)
