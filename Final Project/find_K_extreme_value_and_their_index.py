import random
num_list=[]
for i in range(10):
  num_list.append(random.randint(0,100))
monotonic=[]
backup=[]
k=3
monotonic.append(num_list[0])
for i in range(1,len(num_list)):
	if(num_list[i]>monotonic[-1]):
		if(len(monotonic)<k):
			monotonic.append(num_list[i])
	elif(num_list[i]<monotonic[-1]):
		if(len(monotonic)>=k):
			del monotonic[-1]
		while(len(monotonic)>0):
			if(monotonic[-1]>num_list[i]):
				backup.append(monotonic[-1])
				del monotonic[-1]
			else:
				break
		monotonic.append(num_list[i])
		while(len(backup)>0):
			monotonic.append(backup[-1])
			del backup[-1]

### print前K小的數字
print(num_list)
print(monotonic)
print(backup)

monotonic=[]
backup=[]
monotonic.append(0)
for i in range(1,len(num_list)):
	if(num_list[i]>num_list[monotonic[-1]]):
		if(len(monotonic)<k):
			monotonic.append(i)
	elif(num_list[i]<num_list[monotonic[-1]]):
		if(len(monotonic)>=k):
			del monotonic[-1]
		while(len(monotonic)>0):
			if(num_list[monotonic[-1]]>num_list[i]):
				backup.append(monotonic[-1])
				del monotonic[-1]
			else:
				break
		monotonic.append(i)
		while(len(backup)>0):
			monotonic.append(backup[-1])
			del backup[-1]

### print前K小的數字的index
print(num_list)
print(monotonic)
print(backup)


# print(num_list)
# from dis import dis
# import random
# from venv import create
# import pylab
# import math
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn import metrics
# from sklearn.metrics import r2_score
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split
# import statistics

# class Passengers(object):
#     def __init__(self, cabinClass, age, gender, firstname, lastname, survived):
#         self.cabinClass = cabinClass
#         self.age = age
#         self.gender = gender
#         self.firstname = firstname
#         self.lastname = lastname
#         self.survived = survived
#         self.iCab = -1
#         self.iAge = -1

#     def get_lable(self):
#         return self.survived
    
#     def get_fetures(self):
#         return self.cabinClass, self.age, self.gender

# def calculate_cor_coefficient(list1,list2):
#     x=np.array(list1)
#     tmp=np.array(list2)
#     y=tmp.astype(float)
#     xm = x.mean()
#     ym = y.mean()
#     numerator = np.sum(((x - xm) * (y - ym)))
#     denominator = np.sqrt(np.sum((x - xm) ** 2)) * np.sqrt(np.sum((y - ym) ** 2))
#     return numerator / denominator

# def iScaleFeatures(vals): 
#     """Assumes vals is a sequence of floats""" 
#     minVal, maxVal = min(vals), max(vals) 
#     fit = pylab.polyfit([minVal, maxVal], [0, 1], 1) 
#     return pylab.polyval(fit, vals) 

# def read_data(filename):
#     data={}
#     data["cabinClass"],data["age"], data["gender"], data["survived"],data["firstname"], data["lastname"] = [],[],[],[],[],[]
#     fin=open(filename)
#     line=fin.readline()
#     line=fin.readline()
#     cor=[]
#     while line:
#         line_list=line.rstrip().split(',')
#         for i in range(4):
#             if i!=2:
#                 line_list[i]=float(line_list[i])
#             else:
#                 if line_list[2] == "M":
#                     line_list[2]=1
#                 else:
#                     line_list[2]=0
#         data["cabinClass"].append(line_list[0])
#         data["age"].append(line_list[1])
#         data["gender"].append(line_list[2]) 
#         data["survived"].append(line_list[3])
#         data["lastname"].append(line_list[4])
#         data["firstname"].append(line_list[5])
#         line=fin.readline()
#     cor.append(calculate_cor_coefficient(data["cabinClass"], data["survived"]))
#     cor.append(calculate_cor_coefficient(data["age"], data["survived"]))
#     cor.append(calculate_cor_coefficient(data["gender"], data["survived"]))
#     print(cor)
#     return cor, data

# def create_object(data):
#     pl=[]
#     for i in range(len(data['cabinClass'])):
#         pl.append(Passengers(data['cabinClass'][i],data['age'][i],data['gender'][i],data['firstname'][i], data['lastname'][i],data['survived'][i]))
#     data['age']=iScaleFeatures(data['age'])
#     data['cabinClass']=iScaleFeatures(data['cabinClass'])
#     return pl

# def split_f(pl, num):
#     train=[]
#     test=[]
#     for i in range(len(pl)):
#         if random.randint(0,10000000000000) <= 10000000000000*num:
#             test.append(pl[i])
#         else:
#             train.append(pl[i])
#     return train, test

# def knn(pl, k, num, cor):
#     train, test=split_f(pl, num)
#     predict=[]
#     for i in test:
#         distance=[]
#         for j in train:
#             distance.append(dist(i,j, cor))
#         myStack=[]
#         for j in range(len(distance)):
#             if len(myStack)<k:
#                 counter=0
#                 for l in range(len(myStack)):
#                     if distance[j][0]>distance[myStack[l]][0]:
#                         break
#                     else:
#                         counter+=1
#                 myStack.insert(counter,j)
#             else:
#                 for l in range(len(myStack)):
#                     if distance[j][0]>distance[myStack[l]][0]:
#                         myStack.insert(l, j)
#                         myStack.pop()
#                         break
#         a=0
#         for j in range(len(myStack)):
#             a+=distance[myStack[j]][1]
#         if a<=(k*0.5):
#             predict.append(0)
#         else:
#             predict.append(1)
#     accuracy=0
#     for i in range(len(test)):
#         if test[i].get_lable()==predict[i]:
#             accuracy += 1
#     accuracy = accuracy/len(test)
#     return accuracy, predict
        
# def dist(test, train, cor):
#     te=test.get_fetures()
#     tr=train.get_fetures()
#     dis=0
#     for i in range(len(cor)):
#         dis+=((te[i]-tr[i])*cor[i])**2
#     return dis, train.get_lable()

# cor, data=read_data("TitanicPassengers.txt")
# pl=create_object(data)
# print(knn(pl, 3, 0.2, cor))
# import random
# import pylab
# import math
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn import metrics
# from sklearn.metrics import r2_score
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split
# import statistics
# from sklearn.neighbors import KNeighborsClassifier
# class Passenger(object):
# 	def __init__ (self,CabinClass,age,gender,survived,name):
# 		self.featureVec=[CabinClass,age,gender]
# 		self.CabinClass=CabinClass
# 		self.age=age
# 		self.gender=gender
# 		self.name=name
# 		self.label=survived
# 	def featureDist(self,other):
# 		dist=0.0
# 		for i in range(len(self.featureVec)):
# 			dist+=abs(self.featureVec[i]-other.featureVec[i])**2
# 		return dist**0.5

# 	def cosine_similarity(self,other):
# 	#compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
# 		sumxx, sumxy, sumyy=0,0,0
# 		for i in range(len(self.featureVec)):
# 			x = self.featureVec[i]; y = other.featureVec[i]
# 			sumxx += x*x
# 			sumyy += y*y
# 			sumxy += x*y
# 		return sumxy/math.sqrt(sumxx*sumyy)

# 	def getLabel(self): 
# 		return self.label 

# 	def getFeatures(self): 
# 		return self.featureVec 
# 	def getCabinClass(self):
# 		return self.CabinClass
# 	def getAge(self):
# 		return self.age
# 	def getGender(self):
# 		return self.gender
# 	def getName(self):
# 		return self.name

# 	def __str__ (self): 
# 		return str(self.getFeatures()) + ', ' + str(self.getLabel())
# def read_file_and_preprocessing(filename):
# 	data={}
# 	f=open(filename)
# 	line=f.readline()
# 	data['CabinClass'],data['age'],data['gender'],data['survived'],data['name']=[],[],[],[],[]
# 	line=f.readline()
# 	while line!='':
# 		split=line.rstrip().split(',')
# 		data['CabinClass'].append(int(split[0]))
# 		data['age'].append(float(split[1]))
# 		if(split[2]=='M'):
# 			data['gender'].append(1)
# 		elif(split[2]=='F'):
# 			data['gender'].append(0)
# 		data['survived'].append(int(split[3]))
# 		data['name'].append(split[4]+split[5])
# 		line=f.readline()
# 	f.close()

# 	return data


# def buildPassengerExamples(filename):
# 	data=read_file_and_preprocessing(filename)
# 	examples=[]
# 	for i in range(len(data['CabinClass'])):
# 		p=Passenger(data['CabinClass'][i],data['age'][i],data['gender'][i],data['survived'][i],data['name'][i])
# 		examples.append(p)
# 	return examples

# examples=buildPassengerExamples('TitanicPassengers.txt')


# X=[]
# y=[]
# for i in range(len(examples)):
# 	X.append(examples[i].getFeatures())
# 	y.append(examples[i].getLabel())
# KnnModel=KNeighborsClassifier(n_neighbors=3)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
# KnnModel.fit(np.array(X_train),np.array(y_train).reshape(-1,1))
# y_predicted= KnnModel.predict(X_test)
# print(y_test)
# print(y_predicted)
# print(metrics.accuracy_score(y_test,y_predicted))