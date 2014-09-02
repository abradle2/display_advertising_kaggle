### Aaron J. Bradley
### Initial preprocessing script for csv files in Kaggle Display Ad Challenge

### This script scales the values in the training data

import csv
import numpy as np
from matplotlib import pyplot as plt
import math
import pickle
from sklearn.feature_selection import RFECV, RFE
from sklearn.svm import SVC, OneClassSVM
from sklearn.decomposition import RandomizedPCA
from sklearn.covariance import EllipticEnvelope

class preprocess():

	def __init__(self, file_names, data="train"):
		## Loads data into python arrays

		HeaderArray = []
		DataArray = []
		
		for file_num in range(len(file_names)):
			i = 0
			with open(file_names[file_num], 'r') as File:
				print "Loading file %s" %file_num
				Reader = csv.reader(File, delimiter = ',')
				for row in Reader:
					i = i+1 #annoying, but i starts at 1, not 0
					#Put headers into trainTargetArray
					if i == 1 and file_num == 0:
						HeaderArray.append(row)
					elif i > 1:
						_row = []
						for item in range(len(row)):
							###CHANGE to 14 for test data, 15 for train data
							if item < 14 and row[item] != '':
								_row.append(int(row[item]))
							elif item >= 14 and row[item] != '':
								_row.append(int(row[item], 16))
							elif row[item] == '':
								_row.append(np.nan)
						DataArray.append(_row)

		# Now put the data into numpy arrays
		print "Converting Header array to numpy array..."
		self.HeaderArray = np.array(HeaderArray)
		print "Converting Data array to numpy array..."
		self.DataArray = np.array(DataArray)

		# TargetArray only makes sense for training data.
		# Ignore for test data
		print "Extracting targets into separate array..."
		self.TargetArray = self.DataArray[:,1]

		if data == "train":
			print "Deleting targets from data array..."
			self.DataArray = np.delete(self.DataArray, [0,1], axis=1)
			print "Shape of Data Array: " + str(self.DataArray.shape)
		if data == "test":
			print "Deleting targets from data array..."
			self.DataArray = np.delete(self.DataArray, 0, axis=1)
			print "Shape of Data Array: " + str(self.DataArray.shape)


		# Replace all missing values with NaN
		#self.trainDataArray[self.trainDataArray==''] = np.nan

	def histograms(self, features, data="train"):
		## For visualizing histograms of the integer features

		print "Creating histograms..."

		num_rows = math.ceil(math.sqrt(len(features)))
		plot_num = 1
		for feature in features:
			plt.subplot(num_rows, num_rows, plot_num)
			_feature = []
			for i in range(len(self.DataArray)):
				if not np.isnan(self.DataArray[i][feature]):
					_feature.append(self.DataArray[i][feature])
			plt.hist(_feature, 50)
			plt.title(self.HeaderArray[0][feature])
			plt.ylabel("freq")
			plot_num = plot_num + 1
		plt.show()

	def remove_nulls(self, data = "train"):
		## Remove all data instances which contain a null value
		## PROBABLY NOT USEFUL

		print "Number of data instances before null removal: "
		print len(_data)

		null_instances = []
		for instance in range(len(self.DataArray)):
			for val in self.DataArray[instance]:
				if not np.isnan(val):
					null_instances.append(instance)
		self.DataArray = np.delete(self.DataArray, np.unique(null_instances), axis=0)
		self.TargetArray = np.delete(self.TargetArray, np.unique(null_instances), axis=0)
		print "Number of data instances after null removal: "
		print len(self.DataArray)


	def remove_null_clickcounts(self, data = "train"):
		## Remove those data instances which don't have a value for
		## the "label" feature (ie the most important feature in data)

		## ALWAYS RUN THIS METHOD 

		print "Number of data instances before null removal: "
		print len(self.TargetArray)

		null_instances = []
		for instance in range(len(self.TargetArray)):
			if np.isnan(self.TargetArray[instance][1]):
					null_instances.append(instance)
		self.DataArray = np.delete(self.DataArray, np.unique(null_instances), axis=0)
		self.TargetArray = np.delete(self.TargetArray, np.unique(null_instances), axis=0)
		print "Number of data instances after null removal: "
		print len(self.TargetArray)
		

	def remove_outliers_sd(self, sd=1, data="train"):
		## Remove a data instance if any of its feature values
		## are more than sd standard deviations from the mean
		## for that feature

		print "Removing outliers..."

		_indices = []
		for feature in range(len(self.DataArray[0])):
			if feature > 0:		#do not include ID column
				_mean = np.nanmean(self.DataArray[:,feature])
				_sd = np.nanstd(self.DataArray[:,feature])

				for instance in range(len(self.DataArray)):
					if self.DataArray[instance][feature] > _mean + (sd * _sd) or self.DataArray[instance][feature] < _mean - (sd * _sd):
						_indices.append(instance)
				print "Feature: " + str(feature)
				print "mean: " + str(_mean)
				print "_sd: " + str(_sd)
				print "mean + sd*_sd: " + str(_mean + sd*_sd) + "\n"
		print "Number of data instances before outlier removal: "
		print len(self.DataArray)
		self.DataArray = np.delete(self.DataArray, np.unique(_indices), axis=0)
		self.TargetArray = np.delete(self.TargetArray, np.unique(_indices), axis=0)
		print "Number of data instances after outlier removal: "
		print len(self.DataArray)


	def normalize_mean(self,data="train"):
		## Normalize the features by (x_i - mean)/sd
		## Gives features with ~0 mean. Best for gradient descent

		print "Normalizing the data..."

		_mean = []
		_sd = []
		for feature in range(len(self.DataArray[0])):
			_mean.append( np.nanmean( self.DataArray[:,feature]) )
			_sd.append( np.nanstd(self.DataArray[:,feature]) )

		for i in range(len(self.DataArray)):
			self.DataArray[i] = (self.DataArray[i] - _mean) / _sd

	def normalize_length(self, data="train"):
		## Normalize the features by (x_i/length), where length is the 
		## length of the vector of data values for the given feature

		print "Normalizing the data..."

		for feature in range(len(self.DataArray[0])):
			if feature > 0:		#do not include ID column
				_length = np.nansum(self.DataArray[:,feature]**2)**0.5
				
				for instance in range(len(self.DataArray)):
					self.DataArray[instance][feature] = self.DataArray[instance][feature] / _length

	def normalize_max(self, data="train"):
		## Normalize the features by (x_i/max(x_i))

		print "Normalizing the data..."

		for feature in range(len(self.DataArray[0])):
			if feature > 0:		#do not include ID column
				_max = np.nanmax(self.DataArray[:,feature])
				
				for instance in range(len(self.DataArray)):
					self.DataArray[instance][feature] = self.DataArray[instance][feature] / _max

		
	def count_nulls(self, data="train"):
		## Count how many instances have null values (ie NaN)
		## for each feature

		for feature in range(len(self.DataArray[0])):
			num_nulls = np.count_nonzero(np.isnan(self.DataArray[:,feature]))
			print "Feature: " + str(feature)
			print num_nulls
			print "\n"


	def replace_nulls_mean(self, data="train"):
		## Replace all null values with the mean for that feature

		print "Replacing null values..."

		for feature in range(len(self.DataArray[0])):
			_mean = np.nanmean(self.DataArray[:,feature])
			_indices = np.where(np.isnan(self.DataArray[:,feature]))
			self.DataArray[_indices, feature] = _mean


	def rfe_elimination(self, data="train"):
		## Uses recursive feature elimination to prune the number
		## of features

		print "Running Recursive Elimination of features..."

		estimator = SVC(C=1, degree=3, gamma=0.0, kernel="rbf")
		selector = RFE(estimator, n_features_to_select=30, step=1, verbose=1)
		selector = selector.fit(self.DataArray, self.TargetArray)
		feature_rank = selector.ranking_
		print "Feature Rank:"
		print feature_rank

		if data == "train":
			_indices_remove = []
			for _index in range(len(feature_rank)):
				if feature_rank[_index] != 1:
					_indices_remove.append(_index+2)
			self.DataArray = np.delete(self.DataArray, _indices_remove, axis=1)
			self.TargetArray = np.delete(self.TargetArray, _indices_remove, axis=1)

	def pca_elimination(self, data="train"):
		## Eliminates features based on a principle component analysis

		print "Starting PCA feature reduction..."
		print "Number of features before PCA: " + str(len(self.DataArray[0]))
		n_components = 20

		pca = RandomizedPCA(n_components=n_components).fit(self.DataArray)
		print "Finished running PCA analysis. Transforming data..."

		self.DataArray = pca.transform(self.DataArray)

		print "Number of features after PCA: " + str(len(self.DataArray[0]))

		# Save pca so we can transform test data later on
		output_file = "pca.pickle"
		output = open(output_file, 'wb')
		pickle.dump(pca, output, pickle.HIGHEST_PROTOCOL)
		output.close()


	def remove_outliers_ee(self):
		## Remove outliers using an elliptic envelope

		ee = EllipticEnvelope(contamination=0.1)
		fit = ee.fit(self.DataArray, self.TargetArray)


	def remove_outliers_SVM(self):
		## Remove outliers using a OneClassSVM method

		print "Running SVM to remove outliers..."

		svm = OneClassSVM(kernel='rbf', nu=0.1, degree=3, verbose=1)
		fit = svm.fit(self.DataArray)
		decision = svm.decision_function(self.DataArray)
		_indices = []

		# If a value is below the decision hyperplane, eliminate it
		for i in range(len(decision)):
			if decision[i] < 0:
				_indices.append(i)
		print self.DataArray.shape
		self.DataArray = np.delete(self.DataArray, _indices, axis=0)
		self.TargetArray = np.delete(self.TargetArray, _indices, axis=0)
		print self.DataArray.shape



	def pickle_arrays(self, file_path="/Users/abradle2/Documents/display_ad_challenge/", name="train", data="train"):
		## Use pickle to store a copy of the data and/or target
		## arrays to disk

		print "Pickling arrays ..."
		
		output_file = file_path + name + "_Data.pickle"
		output = open(output_file, 'wb')
		pickle.dump(self.DataArray, output, pickle.HIGHEST_PROTOCOL)
		output.close()

		output_file = file_path + name + "_Targets.pickle"
		output = open(output_file, 'wb')
		pickle.dump(self.TargetArray, output, pickle.HIGHEST_PROTOCOL)
		output.close()

	def load_train_targets(self, file_name):
		## Loads the targets array saved by pickling the training procedure
		## This is critical to determine which features to eliminate
		## in the test data

		train_targets = pickle.load( open(file_name, "rb") )
		print train_targets

	def eliminate_features_manually(self):
		## Manually eliminate features. Useful for forming the test
		## dataset

		_indices = [1,2,4,7,14,15,16,19,21,23,24,25,27,29,33,34,36,38,39]

		self.DataArray = np.delete(self.DataArray, _indices, axis=1)
		self.TargetArray = np.delete(self.TargetArray, _indices, axis=1)


file_path = "/Users/abradle2/Documents/display_ad_challenge/train/"
file_names = []
for i in range(1,101):
	file_names.append(file_path + "train_%s.csv" %i)

train1 = preprocess(file_names)
#train1.count_nulls()
train1.replace_nulls_mean()
train1.pca_elimination()
train1.normalize_mean()

train1.remove_outliers_SVM()
#train1.remove_outliers_sd(sd=3)
#train1.rfe_elimination()
train1.pickle_arrays()
#train1.histograms(range(1,15))


'''
file_name_test = "/Users/abradle2/Documents/display_ad_challenge/test.csv"
test1 = preprocess(file_name_test, data="test")
test1.replace_nulls_mean()
test1.pca_elimination()
test1.normalize_mean()
test1.remove_outliers_SVM()
test1.pickle_arrays()
#train1.histograms(range(1,15))
'''


