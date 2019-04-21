#!/home/brenda/anaconda2/bin/python

import sys
import optparse
import numpy as np
from sklearn import tree
import random
import matplotlib.pyplot as plt

#pre-process data
def processText(inputd, data):
	rawdata = inputd.strip()
	fields = inputd.split(',')

	#convert gender into numeric value, male = 1, female = 0
	if fields[1] == 'M':
		fields[1] = 1
	else:
		fields[1] = 0

	#convert age into integers
	fields[2] = int(fields[2])

	#convert marital status into numeric value, Married = 0, single = 1
	if fields[3] == 'Married':
		fields[3] = 0
	else:
		fields[3] = 1

	#convert current plan into numeric value, low = 1, medium = 2, heavy = 3, prepaid = 4
	if fields[4] == 'Low':
		fields[4] = 1
	elif fields[4] == 'Medium':
		fields[4] = 2
	elif fields[4] == 'Heavy':
		fields[4] = 3
	else:
		fields[4] = 4

	#convert payment methond into numeric values, automatic = 0, non-automatic = 1
	if fields[5] == 'Automatic':
		fields[5] = 0
	else:
		fields[5] = 1

	# convert contract length into numeric values, no contract = 0, 12 = 1, 24 = 2, 36 = 3
	if fields[6] == '12 Months':
		fields[6] = 1
	elif fields[6] == '24 months':
		fields[6] = 2
	elif fields[6] == '36 Months':
		fields[6] = 3
	else: 
		fields[6] = 0

	#convert has kids into numeric values, yes = 1, no = 0
	if fields[7] == 'Y':
		fields[7] = 1
	else:
		fields[7] = 0
	#convert other service into numeric values, yes = 1, no = 0
	if fields[8] == 'Y':
		fields[8] = 1
	else:
		fields[8] = 0

	#convert adopter class into numeric values, very late = 1, else = 0
	if fields[9] == 'Very Late\r\n':
		fields[9] = 1
	else:
		fields[9] = 0
 
	#remove ID from list
	fields.remove(fields[0])
	data.append(fields)
	print "behold, an ordinary python list: ", fields
	
#create a new option parser
parser = optparse.OptionParser()
#add an option to look for the -f 
parser.add_option('-f', '--file', dest='fileName', help='file name to read from')
#get the options entered by the user at the terminal
(options, others) = parser.parse_args()

usingFile = False
#inspect the options entered by the user!
if options.fileName is None:
	print "DEBUG: the user did not enter the -f option"
else:
	print "DEBUG: the user entered the -f option"
	usingFile = True
if(usingFile == True):
	#attempt to open and read out of the file
	print "DEBUG: the file name entered was: ", options.fileName
	file = open(options.fileName, "r") # "r" means we are opening the file for reading
	data = []
	#write a loop that will read one line from the file at a time..
	for line in file:
		processText(line, data)
else:
	#read from standard input 
	print "DEBUG: will read from standard input instead"
	for line in sys.stdin.readlines():
        	processText(line) 

data = np.array(data)

def split(arr, cond):
  return [arr[cond], arr[~cond]]

#separate data into binary target classes: verylate, notverylate
newdata = split(data, data[:,8]==1)
newdata = np.array(newdata)
verylate = newdata[0]
notverylate = newdata[1]

#shuffle data
random_index = random.sample(range(0,notverylate.shape[0]),verylate.shape[0])
newnotvl = notverylate[[random_index]]
prunedata = np.concatenate((verylate, newnotvl), axis=0)
np.random.shuffle(prunedata)

#training dataset
trainingX = []
trainingY = []

#test dataset
testX = [] 
testY = [] 

for i in range(0, prunedata.shape[0]): 
	#construct a single new row as a list for 8 features
	newrow = []
	for n in range(0, prunedata.shape[1]-1):			
		newrow.append(prunedata[i][n])
	#let's split data samples between training and test sets (lists)
	#my approach is to take every tenth data point to generate a training:test of 90:10
	if(i % 10 == 0): #if i is evenly divisible by 10
	#take every tenth record (row) and put it in the test set
		testX.append(newrow)
		testY.append(prunedata[i][8])
	else:
		trainingX.append(newrow)
		trainingY.append(data[i][8])

#the last step in data processing/preparation is to convert the training
#sets to be numpy arrays for efficient processing by scikit
trainingX = np.array(trainingX)
trainingY = np.array(trainingY)
#also convert testX and testY to numpy arrays
testX = np.array(testX)
testY = np.array(testY)

#train a decision tree classifier
#create a new model with some default meta-parameters
maxdepth = [None, 2, 4, 8, 16]
depth = 0
while depth in range(0,len(maxdepth)):
	maxleafnodes = [2, 4, 8, 16, 32, 64, 128, 256]
	accuracylist = []
	for m in maxleafnodes:
		if depth == 0:		
			clf = tree.DecisionTreeClassifier(max_leaf_nodes=m)
		else: 
			clf = tree.DecisionTreeClassifier(max_depth=maxdepth[depth], max_leaf_nodes=m)
		#train a decision tree classifier
		print "training the classifier with the training set"
		clf.fit(trainingX, trainingY)

		#evaluate the performance of model
		correct = 0
		incorrect = 0
		#feed the test features through the model, to see how well the model
		#predicts the class from the samples in the test set that it has never seen
		predictions = clf.predict(testX)

		for i in range(0, predictions.shape[0]):
			if (predictions[i] == testY[i]):
				correct += 1
			else:
				incorrect += 1
		print "correct prediction: ", correct, " incorrect predictions: ", incorrect
		
		#compute accuracy
		accuracy = float(correct) / (correct + incorrect)
		print "Model accuracy: ", accuracy
		accuracylist.append(accuracy)

	print "max depth is ", maxdepth[depth]	
	
	if depth == 0:
		plt.title('Max Depth: None')	
	else: 
		plt.title('Max Depth: %f' %maxdepth[depth])
	
	plt.scatter(maxleafnodes, accuracylist, color = 'blue')
	plt.plot(maxleafnodes, accuracylist, color = 'red', lw=2)
	plt.ylabel('Accuracy')
	plt.xlabel('Max Leaf Nodes')
	plt.show()
	
	depth += 1






		

















