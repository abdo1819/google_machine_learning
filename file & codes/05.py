from scipy.spatial import distance
def euc(a,b):
	
	return  distance.euclidean(a,b)

class scrapyKNN():
    def fit(self,x_data,y_data):
        self.x_data= x_data
        self.y_data= y_data

    def predict(self,x_test):
        predictions=[]
        for row in x_test:
            label=self.closest(row)
            predictions.append(label)
            
        return predictions

    def closest(self,row):
    	best_dist  = euc(self.x_data[0],row)
    	best_index = 0
    	for i in range(1,len(self.x_data)):
    		dist  = euc(row,self.x_data[i])
    		if dist<best_dist:
    			best_dist = dist
    			best_index = i
    	return self.y_data[best_index]		

from sklearn import datasets
from sklearn.datasets import load_iris
import random
iris = load_iris()


x= iris.data
y= iris.target


from sklearn.cross_validation import train_test_split
 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .5)

##from sklearn import tree
##clf = tree.DecisionTreeClassifier()

# from sklearn.neighbors import KNeighborsClassifier
clf = scrapyKNN()
clf.fit(x_train,y_train)

prediction= clf.predict(x_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test,prediction))

