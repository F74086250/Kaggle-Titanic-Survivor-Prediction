from cmath import nan
import random
import pylab
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statistics
from scipy import stats

class Passenger(object):
	def __init__ (self,CabinClass,age,gender,survived,name,age_zScale,age_iScale):
		self.featureVec=[CabinClass,age,gender]
		self.featureVec1=[]
		self.featureVec_age_zScale=[]
		self.featureVec_age_iScale=[]
		if CabinClass==1:
			self.featureVec1=[1,0,0,age,gender]
			self.featureVec_age_zScale=[1,0,0,age_zScale,gender]
			self.featureVec_age_iScale=[1,0,0,age_iScale,gender]
		elif CabinClass==2:
			self.featureVec1=[0,1,0,age,gender]
			self.featureVec_age_zScale=[0,1,0,age_zScale,gender]
			self.featureVec_age_iScale=[0,1,0,age_iScale,gender]
		elif CabinClass==3:
			self.featureVec1=[0,0,1,age,gender]
			self.featureVec_age_zScale=[0,0,1,age_zScale,gender]
			self.featureVec_age_iScale=[0,0,1,age_iScale,gender]

		self.CabinClass=CabinClass
		self.age=age
		self.gender=gender
		self.name=name
		self.label=survived
	def featureDist(self,other):
		dist=0.0
		for i in range(len(self.featureVec)):
			dist+=abs(self.featureVec[i]-other.featureVec[i])**2
		return dist**0.5

	def cosine_similarity(self,other):
	#compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
		sumxx, sumxy, sumyy=0,0,0
		for i in range(len(self.featureVec)):
			x = self.featureVec[i]; y = other.featureVec[i]
			sumxx += x*x
			sumyy += y*y
			sumxy += x*y
		return sumxy/math.sqrt(sumxx*sumyy)

	def getLabel(self): 
		return self.label 

	def getFeatures(self): 
		return self.featureVec
	def getFeatures1(self):
		return self.featureVec1 
	def getFeatures_zScale(self):
		return self.featureVec_age_zScale
	def getFeatures_iScale(self):
		return self.featureVec_age_iScale
	def getCabinClass(self):
		return self.CabinClass
	def getAge(self):
		return self.age
	def getGender(self):
		return self.gender
	def getName(self):
		return self.name

	def __str__ (self): 
		return str(self.getFeatures()) + ', ' + str(self.getLabel())

def zScaleFeatures(vals): 
    """Assumes vals is a sequence of floats""" 
    result = pylab.array(vals) 
    mean = sum(result)/len(result) 
    result = result - mean 
    return result/statistics.pstdev(result) 
def iScaleFeatures(vals): 
	"""Assumes vals is a sequence of floats""" 
	minVal, maxVal = min(vals), max(vals) 
	fit = pylab.polyfit([minVal, maxVal], [0, 1], 1) 
	return pylab.polyval(fit, vals) 
def getR2(measure_val,predicted_val):
	leng=len(measure_val)
	sum_up=0.0
	sum_down=0.0
	mean=sum(measure_val)/leng
	for i in range(len(measure_val)):
		sum_up+=(measure_val[i]-predicted_val[i])**2
		sum_down+=(measure_val[i]-mean)**2
	return round((1-(sum_up/sum_down)),4)
def getRMSD(measure_val,predicted_val):
	leng=len(measure_val)
	total=0.0
	for i in range(len(measure_val)):
		total+=((measure_val[i]-predicted_val[i])**2)
	return math.sqrt(total/leng)  
def read_file_and_preprocessing(filename):
	data={}
	f=open(filename)
	line=f.readline()
	data['CabinClass'],data['age'],data['gender'],data['survived'],data['name']=[],[],[],[],[]
	line=f.readline()
	while line!='':
		split=line.rstrip().split(',')
		# if split[0]=='1':
		# 	data['CabinClass'].append((1,0,0))
		# elif split[0]=='2':
		# 	data['CabinClass'].append((0,1,0))
		# elif split[0]=='3':
		# 	data['CabinClass'].append((0,0,1))
		data['CabinClass'].append(int(split[0]))
		data['age'].append(float(split[1]))
		if(split[2]=='M'):
			data['gender'].append(1)
		elif(split[2]=='F'):
			data['gender'].append(0)
		data['survived'].append(int(split[3]))
		data['name'].append(split[4]+split[5])
		line=f.readline()
	age_zScale=[]
	age_iScale=[]
	age_zScale=zScaleFeatures(data['age'])
	age_iScale=iScaleFeatures(data['age'])
	f.close()

	return data,age_zScale,age_iScale

		


def buildPassengerExamples(filename):
	data,zScale,iScale=read_file_and_preprocessing(filename)
	examples=[]
	for i in range(len(data['CabinClass'])):
		p=Passenger(data['CabinClass'][i],data['age'][i],data['gender'][i],data['survived'][i],data['name'][i],zScale[i],iScale[i])
		examples.append(p)
	return examples


def plot_all_and_survived_by_age(data_list_all,data_list_survived,gender):
	fig,ax=plt.subplots()
	ax.hist(data_list_all,bins=20,ec='black',label=f"All {gender} Passengers\nMean:{round(np.mean(np.array(data_list_all)),2)} SD:{round(np.std(np.array(data_list_all)),2)}")
	ax.hist(data_list_survived,bins=20,ec='black',label=f"All {gender} Passengers\nMean:{round(np.mean(np.array(data_list_survived)),2)} SD:{round(np.std(np.array(data_list_survived)),2)}")
	ax.set_xlabel(f'{gender} Ages')
	ax.set_ylabel(f'Number of {gender} Passengers')
	ax.set_title(f'{gender} Passengers and Survived')
	ax.legend()
	return
def plot_all_and_survived_by_CabinClass(data_list_all,data_list_survived,gender):
	fig,ax=plt.subplots()
	values=[0,1,2,3,4]
	plt.xticks(values)
	plt.xticks(values)
	ax.hist(data_list_all,bins=3,ec='black')
	ax.hist(data_list_survived,bins=3,ec='black')
	ax.set_ylabel(f'Number of {gender} Passengers')
	ax.set_title(f'{gender} Cabin Classes and Survived')
	return	
def survivedofPassengers(examples):
	survived_num,dead_num=0,0
	for r in examples:
			if r.getLabel()==1:
				survived_num+=1
			else:
					dead_num+=1
	return survived_num,dead_num

def divide80_20(examples_list): 
    sampleIndices = random.sample(range(len(examples_list)), len(examples_list)//5) 
    trainingSet, testSet = [], [] 
    for i in range(len(examples_list)): 
        if i in sampleIndices: 
            testSet.append(examples_list[i]) 
        else: trainingSet.append(examples_list[i]) 
    return trainingSet, testSet 
def applyModel(model, testSet, label, prob):
	#Create vector containing feature vectors for all test examples
	testFeatureVecs = [e.getFeatures1() for e in testSet]
	probs = model.predict_proba(testFeatureVecs)
	truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
	for i in range(len(probs)):
		if probs[i][1] > prob:
			if testSet[i].getLabel() == label:
				truePos += 1
			else:
				falsePos += 1
		else:
			if testSet[i].getLabel() != label:
				trueNeg += 1
			else:
				falseNeg += 1
	return truePos, falsePos, trueNeg, falseNeg
def applyModel_age_zScale(model, testSet, label, prob):
	#Create vector containing feature vectors for all test examples
	testFeatureVecs = [e.getFeatures_zScale() for e in testSet]
	probs = model.predict_proba(testFeatureVecs)
	truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
	for i in range(len(probs)):
		if probs[i][1] > prob:
			if testSet[i].getLabel() == label:
				truePos += 1
			else:
				falsePos += 1
		else:
			if testSet[i].getLabel() != label:
				trueNeg += 1
			else:
				falseNeg += 1
	return truePos, falsePos, trueNeg, falseNeg
def applyModel_age_iScale(model, testSet, label, prob):
	#Create vector containing feature vectors for all test examples
	testFeatureVecs = [e.getFeatures_iScale() for e in testSet]
	probs = model.predict_proba(testFeatureVecs)
	truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
	for i in range(len(probs)):
		if probs[i][1] > prob:
			if testSet[i].getLabel() == label:
				truePos += 1
			else:
				falsePos += 1
		else:
			if testSet[i].getLabel() != label:
				trueNeg += 1
			else:
				falseNeg += 1
	return truePos, falsePos, trueNeg, falseNeg
def confidence_interval(alpha,datalist):
	data_mean=statistics.mean(datalist)
	data_std=statistics.pstdev(datalist)
	if(data_mean==0 and data_std==0):
		return 0.0
	interval=stats.norm.interval(alpha,data_mean,data_std)
	ans=round((interval[1]-interval[0])/2,3)
	return ans
def accuracy(truePos, falsePos, trueNeg, falseNeg): 
    numerator = truePos + trueNeg 
    denominator = truePos + trueNeg + falsePos + falseNeg 
    return numerator/denominator 
def sensitivity(truePos, falseNeg): 
	try: 
		return truePos/(truePos + falseNeg) 
	except ZeroDivisionError:
		#print(f"Errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr!,sensitivity,{truePos},{falseNeg}") 
		return float("nan")
def specificity(trueNeg, falsePos): 
	try: 
		return trueNeg/(trueNeg + falsePos) 
	except ZeroDivisionError:
		#print(f"Errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr!,specificity,{trueNeg},{falsePos}")  
		return float("nan") 
def posPredVal(truePos, falsePos): 
	try:
		return truePos/(truePos + falsePos) 
	except ZeroDivisionError:
		#print(f"Errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr!,posPredVal,{truePos},{falsePos}")  
		return float("nan")
def negPredVal(trueNeg, falseNeg): 
	try: 
		return trueNeg/(trueNeg + falseNeg) 
	except ZeroDivisionError:
		#print(f"Errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr!,negPredVal,{trueNeg},{falseNeg}")  
		return float("nan") 

def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint = True): 
	accur = accuracy(truePos, falsePos, trueNeg, falseNeg) 
	sens = sensitivity(truePos, falseNeg) 
	spec = specificity(trueNeg, falsePos) 
	ppv = posPredVal(truePos, falsePos) 
	# if toPrint: 
	#     print(' Accuracy =', round(accur, 3)) 
	#     print(' Sensitivity =', round(sens, 3)) 
	#     print(' Specificity =', round(spec, 3)) 
	#     print(' Pos. Pred. Val. =', round(ppv, 3)) 
	return accur, sens, spec, ppv 
def confusionMatrix(truePos, falsePos, trueNeg, falseNeg):
#    print('\nk = ', k)
	print('TP,FP,TN,FN = ', truePos, falsePos, trueNeg, falseNeg)
	print('                     ', 'TP', '\t', 'FP')
	print('Confusion Matrix is: ', truePos,'\t',falsePos)
	print('                     ', trueNeg,'\t',falseNeg)
	print('                     ', 'TN', '\t', 'FN')    
	acc,sen,spec,ppv=getStats(truePos, falsePos, trueNeg, falseNeg)
	print(f"Accuracy = {round(acc,3)}")
	print(f"Sensitivity = {round(sen,3)}")
	print(f"Specificity = {round(spec,3)}")
	print(f"Pos. Pred. Val. = {round(ppv,3)}")
	return acc
    

def buildROC(model, testSet, label, title, plot = True):
    xVals, yVals = [], []
    p = 0.0
    while p <= 1.0:
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet, label, p)
        xVals.append(1.0 - specificity(trueNeg, falsePos))
        yVals.append(sensitivity(truePos, falseNeg))
        p += 0.01
    auroc = metrics.auc(xVals, yVals)
    if plot:
        pylab.figure()
        pylab.plot(xVals, yVals)
        pylab.plot([0,1], [0,1,], '--')
        pylab.title(title +  ' (AUROC = ' + str(round(auroc, 3)) + ')')
        pylab.xlabel('1 - Specificity - False Postitive Rate')
        pylab.ylabel('Sensitivity-True Positive Rate')
        pylab.show()
    return auroc
def buildROC_age_zScale(model, testSet, label, title, plot = True):
    xVals, yVals = [], []
    p = 0.0
    while p <= 1.0:
        truePos, falsePos, trueNeg, falseNeg = applyModel_age_zScale(model, testSet, label, p)
        xVals.append(1.0 - specificity(trueNeg, falsePos))
        yVals.append(sensitivity(truePos, falseNeg))
        p += 0.01
    auroc = metrics.auc(xVals, yVals)
    if plot:
        pylab.figure()
        pylab.plot(xVals, yVals)
        pylab.plot([0,1], [0,1,], '--')
        pylab.title(title +  ' (AUROC = ' + str(round(auroc, 3)) + ')')
        pylab.xlabel('1 - Specificity - False Postitive Rate')
        pylab.ylabel('Sensitivity-True Positive Rate')
        pylab.show()
    return auroc
def buildROC_age_iScale(model, testSet, label, title, plot = True):
    xVals, yVals = [], []
    p = 0.0
    while p <= 1.0:
        truePos, falsePos, trueNeg, falseNeg = applyModel_age_iScale(model, testSet, label, p)
        xVals.append(1.0 - specificity(trueNeg, falsePos))
        yVals.append(sensitivity(truePos, falseNeg))
        p += 0.01
    auroc = metrics.auc(xVals, yVals)
    if plot:
        pylab.figure()
        pylab.plot(xVals, yVals)
        pylab.plot([0,1], [0,1,], '--')
        pylab.title(title +  ' (AUROC = ' + str(round(auroc, 3)) + ')')
        pylab.xlabel('1 - Specificity - False Postitive Rate')
        pylab.ylabel('Sensitivity-True Positive Rate')
        pylab.show()
    return auroc
def plot_maximum_accuracies(data_list):
	fig,ax=plt.subplots()
	ax.hist(data_list,bins=20,color='blue', ec='black',label=f"Mean:{round(np.mean(np.array(data_list)),2)}\nSD:{round(np.std(np.array(data_list)),2)}")
	ax.set_xlabel('Maximum Accuracies')
	ax.set_ylabel('Numbers of Maximum Accuracies')
	ax.set_title('Maximum Accuracies')
	ax.legend()
	plt.show()

def Generate_data_that_plot_the_answer_to_question_3(model,testSet,k_start,k_end):
	allAccuracy=[]
	count=0
	maxAccuracy=0
	maxcount=0
	maxk=0
	for k in range(k_start, k_end, 1):
		truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet, 1, k/1000)
#    print('k=',k/1000, 'yields: ',truePos, falsePos, trueNeg, falseNeg, '\n') 
		accur=accuracy(truePos, falsePos, trueNeg, falseNeg) 
		allAccuracy.append(accur)
		if maxAccuracy < accur:
			maxAccuracy = accur
			maxcount=count
			maxk=k/1000
		count+=1
	return maxAccuracy,maxk,allAccuracy
def Generate_data_that_plot_the_answer_to_question_4_zScale(model,testSet,k_start,k_end):
	allAccuracy=[]
	count=0
	maxAccuracy=0
	maxcount=0
	maxk=0
	for k in range(k_start, k_end, 1):
		truePos, falsePos, trueNeg, falseNeg = applyModel_age_zScale(model, testSet, 1, k/1000)
#    print('k=',k/1000, 'yields: ',truePos, falsePos, trueNeg, falseNeg, '\n') 
		accur=accuracy(truePos, falsePos, trueNeg, falseNeg) 
		allAccuracy.append(accur)
		if maxAccuracy < accur:
			maxAccuracy = accur
			maxcount=count
			maxk=k/1000
		count+=1
	return maxAccuracy,maxk,allAccuracy
def Generate_data_that_plot_the_answer_to_question_4_iScale(model,testSet,k_start,k_end):
	allAccuracy=[]
	count=0
	maxAccuracy=0
	maxcount=0
	maxk=0
	for k in range(k_start, k_end, 1):
		truePos, falsePos, trueNeg, falseNeg = applyModel_age_iScale(model, testSet, 1, k/1000)
#    print('k=',k/1000, 'yields: ',truePos, falsePos, trueNeg, falseNeg, '\n') 
		accur=accuracy(truePos, falsePos, trueNeg, falseNeg) 
		allAccuracy.append(accur)
		if maxAccuracy < accur:
			maxAccuracy = accur
			maxcount=count
			maxk=k/1000
		count+=1
	return maxAccuracy,maxk,allAccuracy
def Q3_and_Q4_plot(all_max_acc,all_best_k,all_k_acc_list,kValues):
	fig,ax=plt.subplots()
	ax.hist(all_max_acc,bins=20,color='blue', ec='black',label=f"Maximum Accuracies\nMean = {round(np.mean(np.array(all_max_acc)),2)} SD = {round(np.std(np.array(all_max_acc)),2)}")
	ax.set_xlabel('Maximum Accuracies')
	ax.set_ylabel('Numbers of Maximum Accuracies')
	ax.set_title('Maximum Accuracies')
	ax.legend()

	fig1,ax1=plt.subplots()
	ax1.hist(all_best_k,bins=20,color='blue', ec='black',label=f"k values for Maximum Accuracies\nMean = {round(np.mean(np.array(all_best_k)),2)} SD = {round(np.std(np.array(all_best_k)),2)}")
	ax1.set_xlabel('Thresholds Values k')
	ax1.set_ylabel('Numbers of ks')
	ax1.set_title('Threshold value k for Maximum Accuracies')
	ax1.legend()
	mean_acc=[]
	maxK=0.0
	maxcount=0

	for i in range(len(kValues)):
		SumAcc=0.0
		for j in range(len(all_k_acc_list)):
			SumAcc+=all_k_acc_list[j][i]
		meanAcc=SumAcc/len(all_k_acc_list)
		mean_acc.append(meanAcc)
		if(meanAcc>mean_acc[maxcount]):
			maxcount=i
			maxK=kValues[i]
			


	fig2,ax2=plt.subplots()
	ax2.plot(kValues, mean_acc,label="Mean Accuracies")
	ax2.plot(maxK, mean_acc[maxcount],'ro',label="Maximum Mean Accuracy")
	ax2.annotate((maxK, round(mean_acc[maxcount],3)), xy=(maxK,mean_acc[maxcount]))
	ax2.set_title('Mean Accuracies for Different Threshold values')
	ax2.set_xlabel('Threshold Values k')
	ax2.set_ylabel('Accuracy')
	ax2.legend()

	plt.show()

def Q5_plot_male(all_max_acc,all_best_k,all_k_acc_list,kValues):
	fig,ax=plt.subplots()
	ax.hist(all_max_acc,bins=20,color='blue', ec='black',label=f"Maximum Accuracies\nMean = {round(np.mean(np.array(all_max_acc)),2)} SD = {round(np.std(np.array(all_max_acc)),2)}")
	ax.set_xlabel('Maximum Accuracies')
	ax.set_ylabel('Numbers of Maximum Accuracies')
	ax.set_title('Male: Maximum Accuracies')
	ax.legend()

	fig1,ax1=plt.subplots()
	ax1.hist(all_best_k,bins=20,color='blue', ec='black',label=f"k values for Maximum Accuracies\nMean = {round(np.mean(np.array(all_best_k)),2)} SD = {round(np.std(np.array(all_best_k)),2)}")
	ax1.set_xlabel('Thresholds Values k')
	ax1.set_ylabel('Numbers of ks')
	ax1.set_title('Male: Threshold value k for Maximum Accuracies')
	ax1.legend()
	mean_acc=[]
	maxK=0.0
	maxcount=0

	for i in range(len(kValues)):
		SumAcc=0.0
		for j in range(len(all_k_acc_list)):
			SumAcc+=all_k_acc_list[j][i]
		meanAcc=SumAcc/len(all_k_acc_list)
		mean_acc.append(meanAcc)
		if(meanAcc>mean_acc[maxcount]):
			maxcount=i
			maxK=kValues[i]
			


	fig2,ax2=plt.subplots()
	ax2.plot(kValues, mean_acc,label="Mean Accuracies")
	ax2.plot(maxK, mean_acc[maxcount],'ro',label="Maximum Mean Accuracy")
	ax2.annotate((maxK, round(mean_acc[maxcount],3)), xy=(maxK,mean_acc[maxcount]))
	ax2.set_title('Male: Mean Accuracies for Different Threshold values')
	ax2.set_xlabel('Threshold Values k')
	ax2.set_ylabel('Accuracy')
	ax2.legend()

	plt.show()
def Q5_plot_female(all_max_acc,all_best_k,all_k_acc_list,kValues):
	fig,ax=plt.subplots()
	ax.hist(all_max_acc,bins=20,color='blue', ec='black',label=f"Maximum Accuracies\nMean = {round(np.mean(np.array(all_max_acc)),2)} SD = {round(np.std(np.array(all_max_acc)),2)}")
	ax.set_xlabel('Maximum Accuracies')
	ax.set_ylabel('Numbers of Maximum Accuracies')
	ax.set_title('Female: Maximum Accuracies')
	ax.legend()

	fig1,ax1=plt.subplots()
	ax1.hist(all_best_k,bins=20,color='blue', ec='black',label=f"k values for Maximum Accuracies\nMean = {round(np.mean(np.array(all_best_k)),2)} SD = {round(np.std(np.array(all_best_k)),2)}")
	ax1.set_xlabel('Thresholds Values k')
	ax1.set_ylabel('Numbers of ks')
	ax1.set_title('Female: Threshold value k for Maximum Accuracies')
	ax1.legend()
	mean_acc=[]
	maxK=0.0
	maxcount=0

	for i in range(len(kValues)):
		SumAcc=0.0
		for j in range(len(all_k_acc_list)):
			SumAcc+=all_k_acc_list[j][i]
		meanAcc=SumAcc/len(all_k_acc_list)
		mean_acc.append(meanAcc)
		if(meanAcc>mean_acc[maxcount]):
			maxcount=i
			maxK=kValues[i]
			


	fig2,ax2=plt.subplots()
	ax2.plot(kValues, mean_acc,label="Mean Accuracies")
	ax2.plot(maxK, mean_acc[maxcount],'ro',label="Maximum Mean Accuracy")
	ax2.annotate((maxK, round(mean_acc[maxcount],3)), xy=(maxK,mean_acc[maxcount]))
	ax2.set_title('Female: Mean Accuracies for Different Threshold values')
	ax2.set_xlabel('Threshold Values k')
	ax2.set_ylabel('Accuracy')
	ax2.legend()

	plt.show()

def findKNearest(example, exampleSet, k):
    kNearest, distances = [], []
    #Build lists containing first k examples and their distances
    for i in range(k):
        kNearest.append(exampleSet[i])
        distances.append(example.featureDist(exampleSet[i]))
    maxDist = max(distances) #Get maximum distance
    #Look at examples not yet considered
    for e in exampleSet[k:]:
        dist = example.featureDist(e)
        if dist < maxDist:
            #replace farther neighbor by this one
            maxIndex = distances.index(maxDist)
            kNearest[maxIndex] = e
            distances[maxIndex] = dist
            maxDist = max(distances)      
    return kNearest, distances
def KNearestClassify(training, testSet, label, k):
    """Assumes training and testSet lists of examples, k an int
       Uses a k-nearest neighbor classifier to predict
         whether each example in testSet has the given label
       Returns number of true positives, false positives,
          true negatives, and false negatives"""
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for e in testSet:
#        nearest, distances = findKNearest(e, training, k)
        nearest, similarities = findKNearest(e, training, k)
        #conduct vote
        numMatch = 0
        for i in range(len(nearest)):
            if nearest[i].getLabel() == label:
                numMatch += 1
        if numMatch > k//2: #guess label
            if e.getLabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else: #guess not label
            if e.getLabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg
def findK(training, minK, maxK, numFolds, label): 
	#Find average accuracy for range of odd values of k 
	accuracies = []
	best_k=0
	max_score=-1.0 
	k_list=[]
	for k in range(minK, maxK + 1, 2): 
			score = 0.0
			k_list.append(k) 
			for i in range(numFolds): #downsample to reduce computation time
					fold = random.sample(training, min(5000, len(training))) 
					examples, testSet = divide80_20(fold) 
					truePos, falsePos, trueNeg, falseNeg = KNearestClassify(examples, testSet, label, k) 
					score += accuracy(truePos, falsePos, trueNeg, falseNeg) 
#            confusionMatrix(truePos, falsePos, trueNeg, falseNeg, k)
			if((score/numFolds)>max_score):
				best_k=k
				max_score=score/numFolds
			accuracies.append(score/numFolds) 
	return accuracies,best_k,max_score,k_list
def Q6_plot(n_fold_cross_validation_list,real_prediction_list,k_list,numFolds):
	fig,ax=plt.subplots()
	ax.plot(k_list,n_fold_cross_validation_list,label=f'n-fold cross validation')
	ax.plot(k_list,real_prediction_list,label=f'Real Prediction')
	ax.set_xlabel('k values for kNN Regression')
	ax.set_ylabel('Accuracy')
	ax.set_title(f"Average Accuracy vs k ({numFolds} folds)")
	ax.legend()
	plt.show()
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
########################################################################## Question 1 ##############################################################################
examples=buildPassengerExamples('TitanicPassengers.txt')
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
########################################################################## Question 2 ##############################################################################

all_male_age=[]
survived_male_age=[]
all_female_age=[]
survived_female_age=[]
all_male_CabinClass=[]
survived_male_CabinClass=[]
all_female_CabinClass=[]
survived_female_CabinClass=[]
for i in range(len(examples)):
	if(examples[i].getGender()==1):
		all_male_age.append(int(examples[i].getAge()))
		all_male_CabinClass.append(int(examples[i].getCabinClass()))
		if(examples[i].getLabel()==1):
			survived_male_age.append(int(examples[i].getAge()))
			survived_male_CabinClass.append(int(examples[i].getCabinClass()))
	elif(examples[i].getGender()==0):
		all_female_age.append(int(examples[i].getAge()))
		all_female_CabinClass.append(int(examples[i].getCabinClass()))
		if(examples[i].getLabel()==1):
			survived_female_age.append(int(examples[i].getAge()))
			survived_female_CabinClass.append(int(examples[i].getCabinClass()))

plot_all_and_survived_by_age(all_male_age,survived_male_age,"Male")
plot_all_and_survived_by_age(all_female_age,survived_female_age,"Female")
plot_all_and_survived_by_CabinClass(all_male_CabinClass,survived_male_CabinClass,"Male")
plot_all_and_survived_by_CabinClass(all_female_CabinClass,survived_female_CabinClass,"Female")
plt.show()

#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
########################################################################## Question 3 ##############################################################################


C1_weight_1000=[]
C2_weight_1000=[]
C3_weight_1000=[]
age_weight_1000=[]
male_gender_weight_1000=[]
accuracy_1000=[]
sensitivity_1000=[]
specificity_1000=[]
pos_pred_val_1000=[]
AUROC_1000=[]
all_maximum_accuracies_1000=[]
all_k_that_lead_to_maximum_accuracy_1000=[]
all_accuracy_produced_by_every_k_1000=[]
kValues=[k/1000 for k in range(400, 601, 1)]


for i in range(1000):
	trainingSet,testSet=divide80_20(examples)
	survived_num,dead_num=survivedofPassengers(trainingSet)
	survived_num_T,dead_num_T=survivedofPassengers(testSet)
	featureVecs,labels=[],[]
	for e in trainingSet:
		featureVecs.append(e.getFeatures1())
		labels.append(e.getLabel())
	model=linear_model.LogisticRegression()
	model.fit(featureVecs,labels)
	C1_weight_1000.append(model.coef_[0][0])
	C2_weight_1000.append(model.coef_[0][1])
	C3_weight_1000.append(model.coef_[0][2])
	age_weight_1000.append(model.coef_[0][3])
	male_gender_weight_1000.append(model.coef_[0][4])
	tp,fp,tn,fn=applyModel(model,testSet,1,0.5)
	acc,sen,spe,pos=getStats(tp,fp,tn,fn,False)
	accuracy_1000.append(acc)
	sensitivity_1000.append(sen)
	specificity_1000.append(spe)
	pos_pred_val_1000.append(pos)
	max_acc,best_k,all_acc_list=Generate_data_that_plot_the_answer_to_question_3(model,testSet,400,601)
	all_maximum_accuracies_1000.append(max_acc)
	all_k_that_lead_to_maximum_accuracy_1000.append(best_k)
	all_accuracy_produced_by_every_k_1000.append(all_acc_list)
	AUROC_1000.append(buildROC(model,testSet,1,"None",False))
	if((i+1)%100==0):
		print(f"1000 trials... ({i+1}/{1000})")


print("\nLogistic Regression:")
print("Averages for all examples 1000 trials with k=0.5")
print(f"Mean weight of C1 = {round(statistics.mean(C1_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C1_weight_1000)}")
print(f"Mean weight of C2 = {round(statistics.mean(C2_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C2_weight_1000)}")
print(f"Mean weight of C3 = {round(statistics.mean(C3_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C3_weight_1000)}")
print(f"Mean weight of age = {round(statistics.mean(age_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,age_weight_1000)}")
print(f"Mean weight of male gender = {round(statistics.mean(male_gender_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,male_gender_weight_1000)}")
print(f"Mean accuracy = {round(statistics.mean(accuracy_1000),3)}, 95% confidence interval = {confidence_interval(0.95,accuracy_1000)}")
print(f"Mean sensitivity = {round(statistics.mean(sensitivity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,sensitivity_1000)}")
print(f"Mean specificity = {round(statistics.mean(specificity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,specificity_1000)}")
print(f"Mean pos. pred. val. = {round(statistics.mean(pos_pred_val_1000),3)}, 95% confidence interval = {confidence_interval(0.95,pos_pred_val_1000)}")
print(f"Mean AUROC = {round(statistics.mean(AUROC_1000),3)}, 95% confidence interval = {confidence_interval(0.95,AUROC_1000)}\n")
Q3_and_Q4_plot(all_maximum_accuracies_1000,all_k_that_lead_to_maximum_accuracy_1000,all_accuracy_produced_by_every_k_1000,kValues)

#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
########################################################################## Question 4 ##############################################################################
##########################################################################  zScaling  ##############################################################################


C1_weight_1000=[]
C2_weight_1000=[]
C3_weight_1000=[]
age_weight_1000=[]
male_gender_weight_1000=[]
accuracy_1000=[]
sensitivity_1000=[]
specificity_1000=[]
pos_pred_val_1000=[]
AUROC_1000=[]
all_maximum_accuracies_1000=[]
all_k_that_lead_to_maximum_accuracy_1000=[]
all_accuracy_produced_by_every_k_1000=[]
kValues=[k/1000 for k in range(400, 601, 1)]


for i in range(1000):
	trainingSet,testSet=divide80_20(examples)
	survived_num,dead_num=survivedofPassengers(trainingSet)
	survived_num_T,dead_num_T=survivedofPassengers(testSet)
	featureVecs,labels=[],[]
	for e in trainingSet:
		featureVecs.append(e.getFeatures_zScale())
		labels.append(e.getLabel())
	model=linear_model.LogisticRegression()
	model.fit(featureVecs,labels)
	C1_weight_1000.append(model.coef_[0][0])
	C2_weight_1000.append(model.coef_[0][1])
	C3_weight_1000.append(model.coef_[0][2])
	age_weight_1000.append(model.coef_[0][3])
	male_gender_weight_1000.append(model.coef_[0][4])
	tp,fp,tn,fn=applyModel_age_zScale(model,testSet,1,0.5)
	acc,sen,spe,pos=getStats(tp,fp,tn,fn,False)
	accuracy_1000.append(acc)
	sensitivity_1000.append(sen)
	specificity_1000.append(spe)
	pos_pred_val_1000.append(pos)
	max_acc,best_k,all_acc_list=Generate_data_that_plot_the_answer_to_question_4_zScale(model,testSet,400,601)
	all_maximum_accuracies_1000.append(max_acc)
	all_k_that_lead_to_maximum_accuracy_1000.append(best_k)
	all_accuracy_produced_by_every_k_1000.append(all_acc_list)
	AUROC_1000.append(buildROC_age_zScale(model,testSet,1,"None",False))
	if((i+1)%100==0):
		print(f"1000 trials... ({i+1}/{1000})")


print("\nLogistic Regression with zScaling:")
print("Averages for all examples 1000 trials with k=0.5")
print(f"Mean weight of C1 = {round(statistics.mean(C1_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C1_weight_1000)}")
print(f"Mean weight of C2 = {round(statistics.mean(C2_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C2_weight_1000)}")
print(f"Mean weight of C3 = {round(statistics.mean(C3_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C3_weight_1000)}")
print(f"Mean weight of age = {round(statistics.mean(age_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,age_weight_1000)}")
print(f"Mean weight of male gender = {round(statistics.mean(male_gender_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,male_gender_weight_1000)}")
print(f"Mean accuracy = {round(statistics.mean(accuracy_1000),3)}, 95% confidence interval = {confidence_interval(0.95,accuracy_1000)}")
print(f"Mean sensitivity = {round(statistics.mean(sensitivity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,sensitivity_1000)}")
print(f"Mean specificity = {round(statistics.mean(specificity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,specificity_1000)}")
print(f"Mean pos. pred. val. = {round(statistics.mean(pos_pred_val_1000),3)}, 95% confidence interval = {confidence_interval(0.95,pos_pred_val_1000)}")
print(f"Mean AUROC = {round(statistics.mean(AUROC_1000),3)}, 95% confidence interval = {confidence_interval(0.95,AUROC_1000)}\n")
Q3_and_Q4_plot(all_maximum_accuracies_1000,all_k_that_lead_to_maximum_accuracy_1000,all_accuracy_produced_by_every_k_1000,kValues)

#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
########################################################################## Question 4 ##############################################################################
##########################################################################  iScaling  ##############################################################################

C1_weight_1000=[]
C2_weight_1000=[]
C3_weight_1000=[]
age_weight_1000=[]
male_gender_weight_1000=[]
accuracy_1000=[]
sensitivity_1000=[]
specificity_1000=[]
pos_pred_val_1000=[]
AUROC_1000=[]
all_maximum_accuracies_1000=[]
all_k_that_lead_to_maximum_accuracy_1000=[]
all_accuracy_produced_by_every_k_1000=[]
kValues=[k/1000 for k in range(400, 601, 1)]


for i in range(1000):
	trainingSet,testSet=divide80_20(examples)
	survived_num,dead_num=survivedofPassengers(trainingSet)
	survived_num_T,dead_num_T=survivedofPassengers(testSet)
	featureVecs,labels=[],[]
	for e in trainingSet:
		featureVecs.append(e.getFeatures_iScale())
		labels.append(e.getLabel())
	model=linear_model.LogisticRegression()
	model.fit(featureVecs,labels)
	C1_weight_1000.append(model.coef_[0][0])
	C2_weight_1000.append(model.coef_[0][1])
	C3_weight_1000.append(model.coef_[0][2])
	age_weight_1000.append(model.coef_[0][3])
	male_gender_weight_1000.append(model.coef_[0][4])
	tp,fp,tn,fn=applyModel_age_iScale(model,testSet,1,0.5)
	acc,sen,spe,pos=getStats(tp,fp,tn,fn,False)
	accuracy_1000.append(acc)
	sensitivity_1000.append(sen)
	specificity_1000.append(spe)
	pos_pred_val_1000.append(pos)
	max_acc,best_k,all_acc_list=Generate_data_that_plot_the_answer_to_question_4_iScale(model,testSet,400,601)
	all_maximum_accuracies_1000.append(max_acc)
	all_k_that_lead_to_maximum_accuracy_1000.append(best_k)
	all_accuracy_produced_by_every_k_1000.append(all_acc_list)
	AUROC_1000.append(buildROC_age_iScale(model,testSet,1,"None",False))
	if((i+1)%100==0):
		print(f"1000 trials... ({i+1}/{1000})")


print("\nLogistic Regression with iScaling:")
print("Averages for all examples 1000 trials with k=0.5")
print(f"Mean weight of C1 = {round(statistics.mean(C1_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C1_weight_1000)}")
print(f"Mean weight of C2 = {round(statistics.mean(C2_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C2_weight_1000)}")
print(f"Mean weight of C3 = {round(statistics.mean(C3_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C3_weight_1000)}")
print(f"Mean weight of age = {round(statistics.mean(age_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,age_weight_1000)}")
print(f"Mean weight of male gender = {round(statistics.mean(male_gender_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,male_gender_weight_1000)}")
print(f"Mean accuracy = {round(statistics.mean(accuracy_1000),3)}, 95% confidence interval = {confidence_interval(0.95,accuracy_1000)}")
print(f"Mean sensitivity = {round(statistics.mean(sensitivity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,sensitivity_1000)}")
print(f"Mean specificity = {round(statistics.mean(specificity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,specificity_1000)}")
print(f"Mean pos. pred. val. = {round(statistics.mean(pos_pred_val_1000),3)}, 95% confidence interval = {confidence_interval(0.95,pos_pred_val_1000)}")
print(f"Mean AUROC = {round(statistics.mean(AUROC_1000),3)}, 95% confidence interval = {confidence_interval(0.95,AUROC_1000)}\n")
Q3_and_Q4_plot(all_maximum_accuracies_1000,all_k_that_lead_to_maximum_accuracy_1000,all_accuracy_produced_by_every_k_1000,kValues)

#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
########################################################################## Question 5 ##############################################################################
########################################################################## No Scaling ##############################################################################
male_examples=[]
female_examples=[]
for i in range(len(examples)):
	if(examples[i].getGender()==1):
		male_examples.append(examples[i])
	elif(examples[i].getGender()==0):
		female_examples.append(examples[i])
C1_weight_1000=[]
C2_weight_1000=[]
C3_weight_1000=[]
age_weight_1000=[]
male_gender_weight_1000=[]
accuracy_1000=[]
sensitivity_1000=[]
specificity_1000=[]
pos_pred_val_1000=[]
AUROC_1000=[]
all_maximum_accuracies_1000=[]
all_k_that_lead_to_maximum_accuracy_1000=[]
all_accuracy_produced_by_every_k_1000=[]
kValues=[k/1000 for k in range(400, 651, 1)]


for i in range(1000):
	trainingSet,testSet=divide80_20(male_examples)
	survived_num,dead_num=survivedofPassengers(trainingSet)
	survived_num_T,dead_num_T=survivedofPassengers(testSet)
	featureVecs,labels=[],[]
	for e in trainingSet:
		featureVecs.append(e.getFeatures1())
		labels.append(e.getLabel())
	model=linear_model.LogisticRegression()
	model.fit(featureVecs,labels)
	C1_weight_1000.append(model.coef_[0][0])
	C2_weight_1000.append(model.coef_[0][1])
	C3_weight_1000.append(model.coef_[0][2])
	age_weight_1000.append(model.coef_[0][3])
	male_gender_weight_1000.append(model.coef_[0][4])
	tp,fp,tn,fn=applyModel(model,testSet,1,0.5)
	#print(f"{i+1}'th trial,{len(testSet)},{tp},{fp},{tn},{fn}")
	acc,sen,spe,pos=getStats(tp,fp,tn,fn,False)
	#print(acc,sen,spe,pos)
	accuracy_1000.append(acc)
	sensitivity_1000.append(sen)
	specificity_1000.append(spe)
	pos_pred_val_1000.append(pos)
	max_acc,best_k,all_acc_list=Generate_data_that_plot_the_answer_to_question_3(model,testSet,400,651)
	all_maximum_accuracies_1000.append(max_acc)
	all_k_that_lead_to_maximum_accuracy_1000.append(best_k)
	all_accuracy_produced_by_every_k_1000.append(all_acc_list)
	AUROC_1000.append(buildROC(model,testSet,1,"None",False))
	if((i+1)%100==0):
		print(f"1000 trials... ({i+1}/{1000})")


print("\nLogistic Regression with Male and Female Separated:")
print("Averages for Male Examples 1000 trials with k=0.5")
print(f"Mean weight of C1 = {round(statistics.mean(C1_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C1_weight_1000)}")
print(f"Mean weight of C2 = {round(statistics.mean(C2_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C2_weight_1000)}")
print(f"Mean weight of C3 = {round(statistics.mean(C3_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C3_weight_1000)}")
print(f"Mean weight of age = {round(statistics.mean(age_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,age_weight_1000)}")
print(f"Mean weight of male gender = {round(statistics.mean(male_gender_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,male_gender_weight_1000)}")
print(f"Mean accuracy = {round(statistics.mean(accuracy_1000),3)}, 95% confidence interval = {confidence_interval(0.95,accuracy_1000)}")
print(f"Mean sensitivity = {round(statistics.mean(sensitivity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,sensitivity_1000)}")
print(f"Mean specificity = {round(statistics.mean(specificity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,specificity_1000)}")
print(f"Mean pos. pred. val. = {round(statistics.mean(pos_pred_val_1000),3)}, 95% confidence interval = {confidence_interval(0.95,pos_pred_val_1000)}")
print(f"Mean AUROC = {round(statistics.mean(AUROC_1000),3)}, 95% confidence interval = {confidence_interval(0.95,AUROC_1000)}\n")
Q5_plot_male(all_maximum_accuracies_1000,all_k_that_lead_to_maximum_accuracy_1000,all_accuracy_produced_by_every_k_1000,kValues)






C1_weight_1000=[]
C2_weight_1000=[]
C3_weight_1000=[]
age_weight_1000=[]
male_gender_weight_1000=[]
accuracy_1000=[]
sensitivity_1000=[]
specificity_1000=[]
pos_pred_val_1000=[]
AUROC_1000=[]
all_maximum_accuracies_1000=[]
all_k_that_lead_to_maximum_accuracy_1000=[]
all_accuracy_produced_by_every_k_1000=[]
kValues=[k/1000 for k in range(400, 651, 1)]


for i in range(1000):
	trainingSet,testSet=divide80_20(female_examples)
	survived_num,dead_num=survivedofPassengers(trainingSet)
	survived_num_T,dead_num_T=survivedofPassengers(testSet)
	featureVecs,labels=[],[]
	for e in trainingSet:
		featureVecs.append(e.getFeatures1())
		labels.append(e.getLabel())
	model=linear_model.LogisticRegression()
	model.fit(featureVecs,labels)
	C1_weight_1000.append(model.coef_[0][0])
	C2_weight_1000.append(model.coef_[0][1])
	C3_weight_1000.append(model.coef_[0][2])
	age_weight_1000.append(model.coef_[0][3])
	male_gender_weight_1000.append(model.coef_[0][4])
	tp,fp,tn,fn=applyModel(model,testSet,1,0.5)
	#print(f"{i+1}'th trial,{len(testSet)},{tp},{fp},{tn},{fn}")
	acc,sen,spe,pos=getStats(tp,fp,tn,fn,False)
	#print(acc,sen,spe,pos)
	accuracy_1000.append(acc)
	sensitivity_1000.append(sen)
	specificity_1000.append(spe)
	pos_pred_val_1000.append(pos)
	max_acc,best_k,all_acc_list=Generate_data_that_plot_the_answer_to_question_3(model,testSet,400,651)
	all_maximum_accuracies_1000.append(max_acc)
	all_k_that_lead_to_maximum_accuracy_1000.append(best_k)
	all_accuracy_produced_by_every_k_1000.append(all_acc_list)
	AUROC_1000.append(buildROC(model,testSet,1,"None",False))
	if((i+1)%100==0):
		print(f"1000 trials... ({i+1}/{1000})")


print("\nLogistic Regression with Male and Female Separated:")
print("Averages for Female Examples 1000 trials with k=0.5")
print(f"Mean weight of C1 = {round(statistics.mean(C1_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C1_weight_1000)}")
print(f"Mean weight of C2 = {round(statistics.mean(C2_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C2_weight_1000)}")
print(f"Mean weight of C3 = {round(statistics.mean(C3_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C3_weight_1000)}")
print(f"Mean weight of age = {round(statistics.mean(age_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,age_weight_1000)}")
print(f"Mean weight of male gender = {round(statistics.mean(male_gender_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,male_gender_weight_1000)}")
print(f"Mean accuracy = {round(statistics.mean(accuracy_1000),3)}, 95% confidence interval = {confidence_interval(0.95,accuracy_1000)}")
print(f"Mean sensitivity = {round(statistics.mean(sensitivity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,sensitivity_1000)}")
print(f"Mean specificity = {round(statistics.mean(specificity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,specificity_1000)}")
print(f"Mean pos. pred. val. = {round(statistics.mean(pos_pred_val_1000),3)}, 95% confidence interval = {confidence_interval(0.95,pos_pred_val_1000)}")
print(f"Mean AUROC = {round(statistics.mean(AUROC_1000),3)}, 95% confidence interval = {confidence_interval(0.95,AUROC_1000)}\n")
Q5_plot_female(all_maximum_accuracies_1000,all_k_that_lead_to_maximum_accuracy_1000,all_accuracy_produced_by_every_k_1000,kValues)





#####################################################################################################################################################################
#####################################################################################################################################################################
########################################################################## Question 5 ##############################################################################
##########################################################################  zScaling  ##############################################################################
C1_weight_1000=[]
C2_weight_1000=[]
C3_weight_1000=[]
age_weight_1000=[]
male_gender_weight_1000=[]
accuracy_1000=[]
sensitivity_1000=[]
specificity_1000=[]
pos_pred_val_1000=[]
AUROC_1000=[]
all_maximum_accuracies_1000=[]
all_k_that_lead_to_maximum_accuracy_1000=[]
all_accuracy_produced_by_every_k_1000=[]
kValues=[k/1000 for k in range(400, 651, 1)]


for i in range(1000):
	trainingSet,testSet=divide80_20(male_examples)
	survived_num,dead_num=survivedofPassengers(trainingSet)
	survived_num_T,dead_num_T=survivedofPassengers(testSet)
	featureVecs,labels=[],[]
	for e in trainingSet:
		featureVecs.append(e.getFeatures1())
		labels.append(e.getLabel())
	model=linear_model.LogisticRegression()
	model.fit(featureVecs,labels)
	C1_weight_1000.append(model.coef_[0][0])
	C2_weight_1000.append(model.coef_[0][1])
	C3_weight_1000.append(model.coef_[0][2])
	age_weight_1000.append(model.coef_[0][3])
	male_gender_weight_1000.append(model.coef_[0][4])
	tp,fp,tn,fn=applyModel_age_zScale(model,testSet,1,0.5)
	#print(f"{i+1}'th trial,{len(testSet)},{tp},{fp},{tn},{fn}")
	acc,sen,spe,pos=getStats(tp,fp,tn,fn,False)
	#print(acc,sen,spe,pos)
	accuracy_1000.append(acc)
	sensitivity_1000.append(sen)
	specificity_1000.append(spe)
	pos_pred_val_1000.append(pos)
	max_acc,best_k,all_acc_list=Generate_data_that_plot_the_answer_to_question_4_zScale(model,testSet,400,651)
	all_maximum_accuracies_1000.append(max_acc)
	all_k_that_lead_to_maximum_accuracy_1000.append(best_k)
	all_accuracy_produced_by_every_k_1000.append(all_acc_list)
	AUROC_1000.append(buildROC_age_zScale(model,testSet,1,"None",False))
	if((i+1)%100==0):
		print(f"1000 trials... ({i+1}/{1000})")


print("\nLogistic Regression with Male and Female Separated with zScaling:")
print("Averages for Male Examples 1000 trials with k=0.5")
print(f"Mean weight of C1 = {round(statistics.mean(C1_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C1_weight_1000)}")
print(f"Mean weight of C2 = {round(statistics.mean(C2_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C2_weight_1000)}")
print(f"Mean weight of C3 = {round(statistics.mean(C3_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C3_weight_1000)}")
print(f"Mean weight of age = {round(statistics.mean(age_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,age_weight_1000)}")
print(f"Mean weight of male gender = {round(statistics.mean(male_gender_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,male_gender_weight_1000)}")
print(f"Mean accuracy = {round(statistics.mean(accuracy_1000),3)}, 95% confidence interval = {confidence_interval(0.95,accuracy_1000)}")
print(f"Mean sensitivity = {round(statistics.mean(sensitivity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,sensitivity_1000)}")
print(f"Mean specificity = {round(statistics.mean(specificity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,specificity_1000)}")
print(f"Mean pos. pred. val. = {round(statistics.mean(pos_pred_val_1000),3)}, 95% confidence interval = {confidence_interval(0.95,pos_pred_val_1000)}")
print(f"Mean AUROC = {round(statistics.mean(AUROC_1000),3)}, 95% confidence interval = {confidence_interval(0.95,AUROC_1000)}\n")
Q5_plot_male(all_maximum_accuracies_1000,all_k_that_lead_to_maximum_accuracy_1000,all_accuracy_produced_by_every_k_1000,kValues)






C1_weight_1000=[]
C2_weight_1000=[]
C3_weight_1000=[]
age_weight_1000=[]
male_gender_weight_1000=[]
accuracy_1000=[]
sensitivity_1000=[]
specificity_1000=[]
pos_pred_val_1000=[]
AUROC_1000=[]
all_maximum_accuracies_1000=[]
all_k_that_lead_to_maximum_accuracy_1000=[]
all_accuracy_produced_by_every_k_1000=[]
kValues=[k/1000 for k in range(400, 651, 1)]


for i in range(1000):
	trainingSet,testSet=divide80_20(female_examples)
	survived_num,dead_num=survivedofPassengers(trainingSet)
	survived_num_T,dead_num_T=survivedofPassengers(testSet)
	featureVecs,labels=[],[]
	for e in trainingSet:
		featureVecs.append(e.getFeatures1())
		labels.append(e.getLabel())
	model=linear_model.LogisticRegression()
	model.fit(featureVecs,labels)
	C1_weight_1000.append(model.coef_[0][0])
	C2_weight_1000.append(model.coef_[0][1])
	C3_weight_1000.append(model.coef_[0][2])
	age_weight_1000.append(model.coef_[0][3])
	male_gender_weight_1000.append(model.coef_[0][4])
	tp,fp,tn,fn=applyModel_age_zScale(model,testSet,1,0.5)
	#print(f"{i+1}'th trial,{len(testSet)},{tp},{fp},{tn},{fn}")
	acc,sen,spe,pos=getStats(tp,fp,tn,fn,False)
	#print(acc,sen,spe,pos)
	accuracy_1000.append(acc)
	sensitivity_1000.append(sen)
	specificity_1000.append(spe)
	pos_pred_val_1000.append(pos)
	max_acc,best_k,all_acc_list=Generate_data_that_plot_the_answer_to_question_4_zScale(model,testSet,400,651)
	all_maximum_accuracies_1000.append(max_acc)
	all_k_that_lead_to_maximum_accuracy_1000.append(best_k)
	all_accuracy_produced_by_every_k_1000.append(all_acc_list)
	AUROC_1000.append(buildROC_age_zScale(model,testSet,1,"None",False))
	if((i+1)%100==0):
		print(f"1000 trials... ({i+1}/{1000})")


print("\nLogistic Regression with Male and Female Separated with zScaling:")
print("Averages for Female Examples 1000 trials with k=0.5")
print(f"Mean weight of C1 = {round(statistics.mean(C1_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C1_weight_1000)}")
print(f"Mean weight of C2 = {round(statistics.mean(C2_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C2_weight_1000)}")
print(f"Mean weight of C3 = {round(statistics.mean(C3_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C3_weight_1000)}")
print(f"Mean weight of age = {round(statistics.mean(age_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,age_weight_1000)}")
print(f"Mean weight of male gender = {round(statistics.mean(male_gender_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,male_gender_weight_1000)}")
print(f"Mean accuracy = {round(statistics.mean(accuracy_1000),3)}, 95% confidence interval = {confidence_interval(0.95,accuracy_1000)}")
print(f"Mean sensitivity = {round(statistics.mean(sensitivity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,sensitivity_1000)}")
print(f"Mean specificity = {round(statistics.mean(specificity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,specificity_1000)}")
print(f"Mean pos. pred. val. = {round(statistics.mean(pos_pred_val_1000),3)}, 95% confidence interval = {confidence_interval(0.95,pos_pred_val_1000)}")
print(f"Mean AUROC = {round(statistics.mean(AUROC_1000),3)}, 95% confidence interval = {confidence_interval(0.95,AUROC_1000)}\n")
Q5_plot_female(all_maximum_accuracies_1000,all_k_that_lead_to_maximum_accuracy_1000,all_accuracy_produced_by_every_k_1000,kValues)



#####################################################################################################################################################################
#####################################################################################################################################################################
########################################################################## Question 5 ##############################################################################
##########################################################################  iScaling  ##############################################################################
C1_weight_1000=[]
C2_weight_1000=[]
C3_weight_1000=[]
age_weight_1000=[]
male_gender_weight_1000=[]
accuracy_1000=[]
sensitivity_1000=[]
specificity_1000=[]
pos_pred_val_1000=[]
AUROC_1000=[]
all_maximum_accuracies_1000=[]
all_k_that_lead_to_maximum_accuracy_1000=[]
all_accuracy_produced_by_every_k_1000=[]
kValues=[k/1000 for k in range(400, 651, 1)]


for i in range(1000):
	trainingSet,testSet=divide80_20(male_examples)
	survived_num,dead_num=survivedofPassengers(trainingSet)
	survived_num_T,dead_num_T=survivedofPassengers(testSet)
	featureVecs,labels=[],[]
	for e in trainingSet:
		featureVecs.append(e.getFeatures1())
		labels.append(e.getLabel())
	model=linear_model.LogisticRegression()
	model.fit(featureVecs,labels)
	C1_weight_1000.append(model.coef_[0][0])
	C2_weight_1000.append(model.coef_[0][1])
	C3_weight_1000.append(model.coef_[0][2])
	age_weight_1000.append(model.coef_[0][3])
	male_gender_weight_1000.append(model.coef_[0][4])
	tp,fp,tn,fn=applyModel_age_iScale(model,testSet,1,0.5)
	#print(f"{i+1}'th trial,{len(testSet)},{tp},{fp},{tn},{fn}")
	acc,sen,spe,pos=getStats(tp,fp,tn,fn,False)
	#print(acc,sen,spe,pos)
	accuracy_1000.append(acc)
	sensitivity_1000.append(sen)
	specificity_1000.append(spe)
	pos_pred_val_1000.append(pos)
	max_acc,best_k,all_acc_list=Generate_data_that_plot_the_answer_to_question_4_iScale(model,testSet,400,651)
	all_maximum_accuracies_1000.append(max_acc)
	all_k_that_lead_to_maximum_accuracy_1000.append(best_k)
	all_accuracy_produced_by_every_k_1000.append(all_acc_list)
	AUROC_1000.append(buildROC_age_iScale(model,testSet,1,"None",False))
	if((i+1)%100==0):
		print(f"1000 trials... ({i+1}/{1000})")


print("\nLogistic Regression with Male and Female Separated with iScaling:")
print("Averages for Male Examples 1000 trials with k=0.5")
print(f"Mean weight of C1 = {round(statistics.mean(C1_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C1_weight_1000)}")
print(f"Mean weight of C2 = {round(statistics.mean(C2_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C2_weight_1000)}")
print(f"Mean weight of C3 = {round(statistics.mean(C3_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C3_weight_1000)}")
print(f"Mean weight of age = {round(statistics.mean(age_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,age_weight_1000)}")
print(f"Mean weight of male gender = {round(statistics.mean(male_gender_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,male_gender_weight_1000)}")
print(f"Mean accuracy = {round(statistics.mean(accuracy_1000),3)}, 95% confidence interval = {confidence_interval(0.95,accuracy_1000)}")
print(f"Mean sensitivity = {round(statistics.mean(sensitivity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,sensitivity_1000)}")
print(f"Mean specificity = {round(statistics.mean(specificity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,specificity_1000)}")
print(f"Mean pos. pred. val. = {round(statistics.mean(pos_pred_val_1000),3)}, 95% confidence interval = {confidence_interval(0.95,pos_pred_val_1000)}")
print(f"Mean AUROC = {round(statistics.mean(AUROC_1000),3)}, 95% confidence interval = {confidence_interval(0.95,AUROC_1000)}\n")
Q5_plot_male(all_maximum_accuracies_1000,all_k_that_lead_to_maximum_accuracy_1000,all_accuracy_produced_by_every_k_1000,kValues)






C1_weight_1000=[]
C2_weight_1000=[]
C3_weight_1000=[]
age_weight_1000=[]
male_gender_weight_1000=[]
accuracy_1000=[]
sensitivity_1000=[]
specificity_1000=[]
pos_pred_val_1000=[]
AUROC_1000=[]
all_maximum_accuracies_1000=[]
all_k_that_lead_to_maximum_accuracy_1000=[]
all_accuracy_produced_by_every_k_1000=[]
kValues=[k/1000 for k in range(400, 651, 1)]


for i in range(1000):
	trainingSet,testSet=divide80_20(female_examples)
	survived_num,dead_num=survivedofPassengers(trainingSet)
	survived_num_T,dead_num_T=survivedofPassengers(testSet)
	featureVecs,labels=[],[]
	for e in trainingSet:
		featureVecs.append(e.getFeatures1())
		labels.append(e.getLabel())
	model=linear_model.LogisticRegression()
	model.fit(featureVecs,labels)
	C1_weight_1000.append(model.coef_[0][0])
	C2_weight_1000.append(model.coef_[0][1])
	C3_weight_1000.append(model.coef_[0][2])
	age_weight_1000.append(model.coef_[0][3])
	male_gender_weight_1000.append(model.coef_[0][4])
	tp,fp,tn,fn=applyModel_age_iScale(model,testSet,1,0.5)
	#print(f"{i+1}'th trial,{len(testSet)},{tp},{fp},{tn},{fn}")
	acc,sen,spe,pos=getStats(tp,fp,tn,fn,False)
	#print(acc,sen,spe,pos)
	accuracy_1000.append(acc)
	sensitivity_1000.append(sen)
	specificity_1000.append(spe)
	pos_pred_val_1000.append(pos)
	max_acc,best_k,all_acc_list=Generate_data_that_plot_the_answer_to_question_4_iScale(model,testSet,400,651)
	all_maximum_accuracies_1000.append(max_acc)
	all_k_that_lead_to_maximum_accuracy_1000.append(best_k)
	all_accuracy_produced_by_every_k_1000.append(all_acc_list)
	AUROC_1000.append(buildROC_age_iScale(model,testSet,1,"None",False))
	if((i+1)%100==0):
		print(f"1000 trials... ({i+1}/{1000})")


print("\nLogistic Regression with Male and Female Separated with iScaling:")
print("Averages for Female Examples 1000 trials with k=0.5")
print(f"Mean weight of C1 = {round(statistics.mean(C1_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C1_weight_1000)}")
print(f"Mean weight of C2 = {round(statistics.mean(C2_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C2_weight_1000)}")
print(f"Mean weight of C3 = {round(statistics.mean(C3_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,C3_weight_1000)}")
print(f"Mean weight of age = {round(statistics.mean(age_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,age_weight_1000)}")
print(f"Mean weight of male gender = {round(statistics.mean(male_gender_weight_1000),3)}, 95% confidence interval = {confidence_interval(0.95,male_gender_weight_1000)}")
print(f"Mean accuracy = {round(statistics.mean(accuracy_1000),3)}, 95% confidence interval = {confidence_interval(0.95,accuracy_1000)}")
print(f"Mean sensitivity = {round(statistics.mean(sensitivity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,sensitivity_1000)}")
print(f"Mean specificity = {round(statistics.mean(specificity_1000),3)}, 95% confidence interval = {confidence_interval(0.95,specificity_1000)}")
print(f"Mean pos. pred. val. = {round(statistics.mean(pos_pred_val_1000),3)}, 95% confidence interval = {confidence_interval(0.95,pos_pred_val_1000)}")
print(f"Mean AUROC = {round(statistics.mean(AUROC_1000),3)}, 95% confidence interval = {confidence_interval(0.95,AUROC_1000)}\n")
Q5_plot_female(all_maximum_accuracies_1000,all_k_that_lead_to_maximum_accuracy_1000,all_accuracy_produced_by_every_k_1000,kValues)





#####################################################################################################################################################################
#####################################################################################################################################################################
########################################################################## Question 6 ##############################################################################
acc=0.0
trainingSet,testSet=divide80_20(examples)
survived_num,dead_num=survivedofPassengers(trainingSet)
survived_num_T,dead_num_T=survivedofPassengers(testSet)
TP,FP,TN,FN=KNearestClassify(trainingSet,testSet,1,3)
print("k-NN Prediction for Survive with k=3:")
acc=confusionMatrix(TP,FP,TN,FN)
minK, maxK, numFolds, label = 1, 25, 10, 1
accuracies,best_k,max_score,k_list= findK(trainingSet, minK, maxK, numFolds, label) 
print("Using n-fold cross validation to find proper k for k-NN Prediction")
print(f"K for Maximum Accuracy is: {best_k}")
trainingSet,testSet=divide80_20(examples)
TP,FP,TN,FN=KNearestClassify(trainingSet,testSet,1,best_k)
acc=confusionMatrix(TP,FP,TN,FN)
print(f"Predictions with maximum accuracy k: {best_k}")
print(f"Cross Validation Accuracies is: {max_score}")
print(f"Predicted Accuracies is: {acc}")
real_predict_k=[]
for i in range(len(k_list)):
	trainingSet,testSet=divide80_20(examples)
	TP,FP,TN,FN=KNearestClassify(trainingSet,testSet,1,k_list[i])
	real_predict_k.append(accuracy(TP,FP,TN,FN))

Q6_plot(accuracies,real_predict_k,k_list,10)




#####################################################################################################################################################################
#####################################################################################################################################################################
########################################################################## Question 7 ##############################################################################
male_examples=[]
female_examples=[]
for i in range(len(examples)):
	if(examples[i].getGender()==1):
		male_examples.append(examples[i])
	elif(examples[i].getGender()==0):
		female_examples.append(examples[i])
all_TP,all_FP,all_TN,all_FN=0,0,0,0
acc=0.0
trainingSet,testSet=divide80_20(male_examples)
survived_num,dead_num=survivedofPassengers(trainingSet)
survived_num_T,dead_num_T=survivedofPassengers(testSet)
TP,FP,TN,FN=KNearestClassify(trainingSet,testSet,1,3)
all_TP+=TP
all_FP+=FP
all_TN+=TN
all_FN+=FN
print("\nTry to predict male and female separately and combined with k=3:")
print("For Male:")
acc=confusionMatrix(TP,FP,TN,FN)

trainingSet,testSet=divide80_20(female_examples)
survived_num,dead_num=survivedofPassengers(trainingSet)
survived_num_T,dead_num_T=survivedofPassengers(testSet)
TP,FP,TN,FN=KNearestClassify(trainingSet,testSet,1,3)
all_TP+=TP
all_FP+=FP
all_TN+=TN
all_FN+=FN

print("\nFor Female:")
acc=confusionMatrix(TP,FP,TN,FN)



print("\nCombined Predictions Statistics:")
acc=confusionMatrix(all_TP,all_FP,all_TN,all_FN)