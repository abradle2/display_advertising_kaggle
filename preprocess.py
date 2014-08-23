### Aaron J. Bradley
### Initial preprocessing script for csv files in Kaggle Display Ad Challenge

### This script scales the values in the training data

import csv
import numpy as np

files = ["/Users/abradle2/Documents/display_ad_challenge/train_1.csv"]

###Load training Data
trainTargetArray = []
trainDataArray = []
i = 0
with open(files[0], 'r') as trainFile:
	trainReader = csv.reader(trainFile, delimiter = ',')
	for row in trainReader:
		i = i+1 #annoying, but i starts at 1, not 0
		#Put headers into trainTargetArray
		if i == 1:
			trainTargetArray.append(row)
			print row
		else:
			trainDataArray.append(row)
print trainDataArray[0]