import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

iris = load_iris()
clf = tree.DecisionTreeClassifier()

list_idx = [0,50,100]
#train data(removing list_idx)
train_data = np.delete(iris.data,list_idx,0)
train_target = np.delete(iris.target,list_idx)
#train
clf = clf.fit(train_data, train_target)
#test
print(iris.target[[0,50,100]])
print(clf.predict(iris.data[[0,50,100]]))

##from sklearn.externals.six import StringIO
##import pydot
##dot_data = StringIO() 
dot_data = tree.export_graphviz(clf,
                     out_file=None,
                     feature_names = iris.feature_names,
                     class_names= iris.target_names,
                     filled=True , rounded=True,
                     impurity=False)

graph = graphviz.Source(dot_data)
graph.render("iris")
####run command ( dot -Tpdf tree.dot -o tree.pdf)
