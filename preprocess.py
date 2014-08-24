### Aaron J. Bradley
### Initial preprocessing script for csv files in Kaggle Display Ad Challenge

### This script scales the values in the training data

import csv
import numpy as np
from matplotlib import pyplot as plt
import math
import pickle
from sklearn.feature_selection import RFECV, RFE
from sklearn.svm import SVC

class preprocess():

	def __init__(self, file_names):
		## Loads data into python arrays

		self.trainTargetArray = []
		self.trainDataArray = []
		
		for file_num in range(len(file_names)):
			i = 0
			with open(file_names[file_num], 'r') as trainFile:
				print "Loading file %s" %file_num
				trainReader = csv.reader(trainFile, delimiter = ',')
				for row in trainReader:
					i = i+1 #annoying, but i starts at 1, not 0
					#Put headers into trainTargetArray
					if i == 1 and file_num == 0:
						self.trainTargetArray.append(row)
					elif i > 1:
						_row = []
						for item in range(len(row)):
							if item < 15 and row[item] != '':
								_row.append(int(row[item]))
							elif item >= 15 and row[item] != '':
								_row.append(int(row[item], 16))
							elif row[item] == '':
								_row.append(np.nan)
						self.trainDataArray.append(_row)

		# Now put the data into numpy arrays
		self.trainTargetArray = np.array(self.trainTargetArray)
		self.trainDataArray = np.array(self.trainDataArray)

		# Replace all missing values with NaN
		#self.trainDataArray[self.trainDataArray==''] = np.nan

	def histograms(self, features, data="train"):
		## For visualizing histograms of the integer features

		print "Creating histograms..."

		if data == "train":
			_data = self.trainDataArray
			_targets = self.trainTargetArray

		num_rows = math.ceil(math.sqrt(len(features)))
		plot_num = 1
		for feature in features:
			plt.subplot(num_rows, num_rows, plot_num)
			_feature = []
			for i in range(len(_data)):
				if not np.isnan(_data[i][feature]):
					_feature.append(_data[i][feature])
			plt.hist(_feature, 50)
			plt.title(_targets[0][feature])
			plt.ylabel("freq")
			plot_num = plot_num + 1
		plt.show()

	def remove_nulls(self, data = "train"):
		## Remove all data instances which contain a null value
		## PROBABLY NOT USEFUL

		if data == "train":
			_data = self.trainDataArray
			_targets = self.trainTargetArray

		print "Number of data instances before null removal: "
		print len(_data)

		null_instances = []
		for instance in range(len(_data)):
			for val in _data[instance]:
				if not np.isnan(val):
					null_instances.append(instance)
		_data = np.delete(_data, np.unique(null_instances), axis=0)
		print "Number of data instances after null removal: "
		print len(_data)
		if data == "train":
			self.trainDataArray = _data

	def remove_null_clickcounts(self, data = "train"):
		## Remove those data instances which don't have a value for
		## the "label" feature (ie the most important feature in data)

		## ALWAYS RUN THIS METHOD 

		if data == "train":
			_data = self.trainDataArray
			_targets = self.trainTargetArray

		print "Number of data instances before null removal: "
		print len(_data)

		null_instances = []
		for instance in range(len(_data)):
			if np.isnan(_data[instance][1]):
					null_instances.append(instance)
		_data = np.delete(_data, np.unique(null_instances), axis=0)
		print "Number of data instances after null removal: "
		print len(_data)
		if data == "train":
			self.trainDataArray = _data

	def remove_outliers_sd(self, features, sd=1, data="train"):
		## Remove a data instance if any of its feature values
		## are more than sd standard deviations from the mean
		## for that feature

		print "Removing outliers..."

		if data == "train":
			_data = self.trainDataArray
			_targets = self.trainTargetArray

		_indices = []
		for feature in features:
			if feature > 0:		#do not include ID column
				_mean = np.nanmean(_data[:,feature])
				_sd = np.nanstd(_data[:,feature])

				for instance in range(len(_data)):
					if _data[instance][feature] > _mean + (sd * _sd) or _data[instance][feature] < _mean - (sd * _sd):
						_indices.append(instance)
				print "Feature: " + str(feature)
				print "mean: " + str(_mean)
				print "_sd: " + str(_sd)
				print "mean + sd*_sd: " + str(_mean + sd*_sd) + "\n"
		print "Number of data instances before outlier removal: "
		print len(_data)
		_data = np.delete(_data, np.unique(_indices), axis=0)
		print "Number of data instances after outlier removal: "
		print len(_data)

		if data == "train":
			self.trainDataArray = _data

	def normalize_mean(self, features=range(2,41),data="train"):
		## Normalize the features by (x_i - mean)/sd
		## Gives features with ~0 mean. Best for gradient descent

		print "Normalizing the data..."

		if data == "train":
			_data = self.trainDataArray
			_targets = self.trainTargetArray		 

		for feature in features:
			_mean = np.nanmean(_data[:,feature])
			_sd = np.nanstd(_data[:,feature])

			for instance in range(len(_data)):
				_data[instance][feature] = (_data[instance][feature] - _mean) / _sd

		if data == "train":
			self.trainDataArray = _data

	def normalize_length(self, features=range(2,41),data="train"):
		## Normalize the features by (x_i/length), where length is the 
		## length of the vector of data values for the given feature

		print "Normalizing the data..."

		if data == "train":
			_data = self.trainDataArray
			_targets = self.trainTargetArray		 

		for feature in features:
			_length = np.nansum(_data[:,feature]**2)**0.5
			

			for instance in range(len(_data)):
				_data[instance][feature] = _data[instance][feature] / _length

		if data == "train":
			self.trainDataArray = _data

	def normalize_max(self, features=range(2,41),data="train"):
		## Normalize the features by (x_i/max(x_i))

		print "Normalizing the data..."

		if data == "train":
			_data = self.trainDataArray
			_targets = self.trainTargetArray		 

		for feature in features:
			_max = np.nanmax(_data[:,feature])
			

			for instance in range(len(_data)):
				_data[instance][feature] = _data[instance][feature] / _max

		if data == "train":
			self.trainDataArray = _data

	def count_nulls(self, features=range(1,41), data="train"):
		## Count how many instances have null values (ie NaN)
		## for each feature

		if data == "train":
			_data = self.trainDataArray
			_targets = self.trainTargetArray

		for feature in features:
			num_nulls = np.count_nonzero(np.isnan(_data[:,feature]))
			print "Feature: " + str(feature)
			print num_nulls
			print "\n"

	def replace_nulls_mean(self, features=range(2,41), data="train"):
		## Replace all null values with the mean for that feature

		print "Replacing null values..."

		if data == "train":
			_data = self.trainDataArray
			_targets = self.trainTargetArray

		for feature in features:
			_mean = np.nanmean(_data[:,feature])
			_indices = np.where(np.isnan(_data[:,feature]))
			_data[_indices, feature] = _mean

		if data == "train":
			self.trainDataArray = _data


	def rfe_elimination(self, data="train"):
		## Uses recursive feature elimination to prune the number
		## of features

		print "Running Recursive Elimination of features..."
		if data == "train":
			X = self.trainDataArray[:,2:]
			y = self.trainDataArray[:,1]
		estimator = SVC(C=1, kernel="linear")
		selector = RFE(estimator, n_features_to_select=20, step=1, verbose=1)
		selector = selector.fit(X,y)
		feature_rank = selector.ranking_
		print "Feature Rank:"
		print feature_rank

		if data == "train":
			_indices_remove = []
			for _index in range(len(feature_rank)):
				if feature_rank[_index] != 1:
					_indices_remove.append(_index+2)
			self.trainDataArray = np.delete(self.trainDataArray, _indices_remove, axis=1)

	def pickle_arrays(self, file_path="/Users/abradle2/Documents/display_ad_challenge/", name="train", data="train"):
		## Use pickle to store a copy of the data and/or target
		## arrays to disk

		print "Pickling arrays ..."
		
		output_file = file_path + name + "_Data.pickle"
		output = open(output_file, 'wb')
		pickle.dump(self.trainDataArray, output, pickle.HIGHEST_PROTOCOL)
		output.close()

		output_file = file_path + name + "_Targets.pickle"
		output = open(output_file, 'wb')
		pickle.dump(self.trainTargetArray, output, pickle.HIGHEST_PROTOCOL)
		output.close()


file_path = "/Users/abradle2/Documents/display_ad_challenge/train/"
file_names = []
for i in range(1,21):
	file_names.append(file_path + "train_%s.csv" %i)
#file_names = ["/Users/abradle2/Documents/display_ad_challenge/train/train_1.csv", "/Users/abradle2/Documents/display_ad_challenge/train/train_2.csv"]

train1 = preprocess(file_names)
train1.count_nulls()
train1.replace_nulls_mean()
train1.remove_outliers_sd(features=[2,3,4,5,6,7,8,9,10,12,13,14], sd=3)
train1.normalize_max()
train1.rfe_elimination()
train1.pickle_arrays()

#train1.histograms(range(2,41))


