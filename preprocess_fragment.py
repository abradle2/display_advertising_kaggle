### Aaron J. Bradley
### Initial preprocessing script for csv files in Kaggle Display Ad Challenge

### This script breaks apart a csv file into multiple smaller
### files for easier manipulation

import csv
#import numpy as np
from os import SEEK_END

#################################################################
#################################################################

##SET THESE VALUES

#Path to csv file
csv_path = '/Users/abradle2/Documents/display_ad_challenge/'

#CSV file name (do not include .csv). We will use this to form 
#the names for the smaller files
file_prefix = 'train'

#Number of lines per smaller file
num_lines = 1000

#Number of smaller files to create (always starting from top of 
#test.csv)
num_files = 100

#################################################################
#################################################################

###Load Test Data
targetArray = []
dataArray = []
i = 0
num_files_written = 0
filename = csv_path + file_prefix + '.csv'
with open(filename, 'r') as _file:
	_reader = csv.reader(_file, delimiter = ',')
	for row in _reader:
		i = i+1 #annoying, but i starts at 1, not 0
		#Put headers into testTargetArray
		if i == 1:
			targetArray.append(row)
			print row
		elif i > 1 and i < num_lines:
			dataArray.append(row)
		elif i == num_lines:
			dataArray.append(row)	#So we don't skip this row
			print "i = %s: writing file %s_%s.csv \n" %(i, file_prefix, num_files_written+1)
			#output targets and data to the fragmented file
			_fragmented_file = csv_path + file_prefix + "/" + file_prefix + "_%s.csv" %(num_files_written+1)
			output = open(_fragmented_file, 'w')
			for target_row in targetArray:
				for target in target_row:
					output.write(str(target) + ",")
				output.seek(-1, SEEK_END)	#to eliminate trailing comma
				output.write("\n")
			for test_row in dataArray:
				for data in test_row:
					output.write(str(data) + ",")
				output.seek(-1, SEEK_END)	#to eliminate trailing comma
				output.write("\n")
			output.close()

			#Reset counter and dataArray
			i = 1
			dataArray = []

			num_files_written = num_files_written + 1
			if num_files_written >= num_files:
				break


