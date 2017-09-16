

from sklearn import tree

#make data
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
#classifier
clf = tree.DecisionTreeClassifier()
clf.fit(features,labels)

print (clf.predict([[145,1]]))

#view in graph
import graphviz
dot_data = tree.export_graphviz(clf,out_file = None
                                ,feature_names=["wieght","smothy"],
                                class_names=["apple","orange"],
                                filled = True,rounded =False,
                                special_characters = True)

graph = graphviz.Source(dot_data)
graph.render("iris")
