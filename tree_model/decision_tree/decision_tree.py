from sklearn.datasets import load_iris
from sklearn import tree

x,y=load_iris(return_X_y=True)
clf=tree.DecisionTreeClassifier()
clf.fit(x,y)
tree.plot_tree(clf)
