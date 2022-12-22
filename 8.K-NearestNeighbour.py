from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
# print("\n IRIS FEATURES \ TARGET NAMES: \n ", iris.target_names)

for i in range(len(iris.target_names)):
    print("\n[{0}]:[{1}]".format(i, iris.target_names[i]))


X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0)

print("\n IRIS DATA :\n", iris.data)
print("\n Target :\n", iris.target)
print("\n X TRAIN \n", X_train)
print("\n X TEST \n", X_test)
print("\n Y TRAIN \n", y_train)
print("\n Y TEST \n", y_test)

kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])
# [Sepal Length, Sepal Width,Petal Length,Petal Width]
print("\n XNEW \n", x_new)

prediction = kn.predict(x_new)
print("\n Predicted target value: {}\n".format(prediction))
print("\n Predicted feature name: {}\n".format(
    iris.target_names[prediction]))

# Now testing the model on the testing dataset
for i in range(len(X_test)):
    # each x is list of values i.e [Sepal Length,Sepal Width,Petal Length,Petal Width]
    # x = X_test[i]
    x_new = np.array([X_test[i]])
    prediction = kn.predict(x_new)
    print("\n Actual : {0} {1}, Predicted :{2}{3}".format(
        y_test[i], iris.target_names[y_test[i]], prediction, iris.target_names[prediction]))
print("\n TEST SCORE[ACCURACY]: {:.2f}\n".format(kn.score(X_test, y_test)))
