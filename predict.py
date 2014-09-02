### Aaron J. Bradley
### Predicting script for Kaggle Display Ad Challenge

import numpy as np
from matplotlib import pyplot as plt
import pickle

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

class predict(object):
	def __init__(self, X_file):
		## Load pickled data

		print "Loading data..."
		data = pickle.load( open(X_file, "rb") )
		self.X = data[:,1:]
		self.IDs = data[:,0]

		print "Length of Data: " + str(len(self.X))

	def predict_random_forest(self, file_name):
		## Run random forest algorithm
		
		print "Loading classifier..."
		clf = pickle.load( open(file_name, "rb") )

		self.prediction = clf.predict_proba(self.X)
		
	def save_prediction(self, file_name):
		## Save a csv file with the prediction probabilities
		
		print "Saving prediction ..."
		output = open(file_name, "w")
		output.write("Id,Predicted\n")
		for i in range(len(self.X)):
			output.write(str(int(self.IDs[i])) + "," + str(self.prediction[i][1]) + "\n")
		
		output.close()



file_path = "/Users/abradle2/Documents/display_ad_challenge/"
data_file = file_path + "test_Data.pickle"


predict_1 = predict(X_file=data_file)
predict_1.predict_random_forest("random_forest_1.pickle")
predict_1.save_prediction("test_prediction.dat")





