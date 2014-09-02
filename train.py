### Aaron J. Bradley
### Processing script for Kaggle Display Ad Challenge

import numpy as np
from matplotlib import pyplot as plt
import pickle

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

class process(object):
	def __init__(self, X_file, y_file):
		## Load pickled data

		print "Loading data..."
		self.X = pickle.load( open(X_file, "rb") )
		self.y = pickle.load( open(y_file, "rb") )

		print "Length of X: " + str(len(self.X))
		print "Length of y: " + str(len(self.y))

	def random_forest(self):
		## Run random forest algorithm
		X = self.X[5000:]
		y = self.y[5000:]

		X_test = self.X[:5000]
		y_test = self.y[:5000]

		print "Running random forest classifier with %i training and %i testing values" %(len(X), len(X_test))

		clf = RandomForestClassifier(n_estimators = 30, verbose=1)
		fit = clf.fit(X,y)
		score = clf.score(X_test, y_test)
		print score

		print "Pickling the classifier ..."
		
		output_file = "random_forest_1.pickle"
		output = open(output_file, 'wb')
		pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
		output.close()

		#scores = cross_val_score(clf, self.X, self.y)
		#print scores.mean()


	def svm(self):
		## Run SVM classifier

		X = self.X[5000:]
		y = self.y[5000:]

		X_test = self.X[:5000]
		y_test = self.y[:5000]

		print "Running SVM classifier with %i training and %i testing values" %(len(X), len(X_test))

		print "Starting SVM classifier"
		clf = SVC(C=1, verbose=1)
		fit = clf.fit(X, y)
		score = clf.score(X_test, y_test)
		print score

		output_file = "svc_1.pickle"
		output = open(output_file, 'wb')
		pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
		output.close()


	def knc(self):
		## Run K nearest neighbors classification

		X = self.X[5000:]
		y = self.y[5000:]

		X_test = self.X[:5000]
		y_test = self.y[:5000]

		print "Running KNN classifier with %i training and %i testing values" %(len(X), len(X_test))

		print "Starting KNN classifier"
		clf = KNC(n_neighbors=100)
		fit = clf.fit(X, y)
		score = clf.score(X_test, y_test)
		print score

		output_file = "knn_1.pickle"
		output = open(output_file, 'wb')
		pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
		output.close()




file_path = "/Users/abradle2/Documents/display_ad_challenge/"
X_file = file_path + "train_Data.pickle"
y_file = file_path + "train_Targets.pickle"


process_1 = process(X_file, y_file)
#process_1.random_forest()
#process_1.svm()
process_1.knc()





