### Aaron J. Bradley

## Scrap python code for one-off computations

import csv
from os import SEEK_END

#################################################################
#################################################################

##SET THESE VALUES

#Path to csv file
csv_path = '/Users/abradle2/Documents/display_ad_challenge/'

#CSV file name (do not include .csv). We will use this to form 
#the names for the smaller files
file_prefix = 'train'

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
		i = i+1
print i

