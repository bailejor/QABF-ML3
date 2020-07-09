import numpy as np
import pandas
from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut
from math import sqrt
from numpy.random import seed
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score




#Attention Function#
################################################################################################################
#Best feature combination was 4-8 (92%)

dataframe = pandas.read_csv("FullTest.csv", header = 0)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X_orig = dataset[:,4:8].astype(float)
y_orig = dataset[:,8:9].astype(float)

loocv = LeaveOneOut()
#Bias toward predicting function present if right probability is high, bias toward absent if left number is high
#Prior probability for attention according to Beavers et al. (2013) was 21.7%
classifier = GaussianNB(priors = [0.783, 0.217])


results_loocv = model_selection.cross_val_score(classifier, X_orig, y_orig.ravel(), cv=loocv)
predict_loocv = model_selection.cross_val_predict(classifier, X_orig, y_orig.ravel(), cv=loocv)
print(results_loocv.mean())
#print(predict_loocv.ravel(), y_orig.ravel())



#Escape Function#
################################################################################################################
#Best feature combination was 4-7 (86%)

dataframe2 = pandas.read_csv("FullTest2.csv", header = 0)
dataset2 = dataframe2.values
# split into input (X) and output (Y) variables
X_orig2 = dataset2[:,4:7].astype(float)
y_orig2 = dataset2[:,8:9].astype(float)

loocv = LeaveOneOut()
#Bias toward predicting function present if right probability is high, bias toward absent if left number is high
#Prior probability for Escape according to Beavers et al. (2013) was 32.2%
classifier2 = GaussianNB(priors = [0.678, 0.322])

results_loocv2 = model_selection.cross_val_score(classifier2, X_orig2, y_orig2.ravel(), cv=loocv)
predict_loocv2 = model_selection.cross_val_predict(classifier2, X_orig2, y_orig2.ravel(), cv=loocv)

print(results_loocv2.mean())
#print(predict_loocv2.ravel(), y_orig2.ravel())



#Non-social Function#
################################################################################################################
#Best feature combination was 4-7 (96%)

dataframe3 = pandas.read_csv("FullTest3.csv", header = 0)
dataset3 = dataframe3.values
# split into input (X) and output (Y) variables
X_orig3 = dataset3[:,4:7].astype(float)
y_orig3 = dataset3[:,8:9].astype(float)
loocv = LeaveOneOut()
#Bias toward predicting function present if right probability is high, bias toward absent if left number is high
#Prior probability for NS according to Beavers et al. (2013) was 16.3%
classifier3 = GaussianNB(priors = [0.837, 0.163])

results_loocv3 = model_selection.cross_val_score(classifier3, X_orig3, y_orig3.ravel(), cv=loocv)
predict_loocv3 = model_selection.cross_val_predict(classifier3, X_orig3, y_orig3.ravel(), cv=loocv)

print(results_loocv3.mean())
#print(predict_loocv3.ravel(), y_orig3.ravel())


#Tangible Function#
################################################################################################################
#Best feature combination was tangible endorse and severity only (90%)
dataframe4 = pandas.read_csv("FullTest4.csv", header = 0)
dataset4 = dataframe4.values
# split into input (X) and output (Y) variables
X_orig4 = dataset4[:,0:2].astype(float)
y_orig4 = dataset4[:,2:3].astype(float)
loocv = LeaveOneOut()
#Bias toward predicting function present if right probability is high, bias toward absent if left number is high
#Prior probability for Tangible according to Beavers et al. (2013) was 11.0%
classifier4 = GaussianNB(priors = [0.89, 0.11])

results_loocv4 = model_selection.cross_val_score(classifier4, X_orig4, y_orig4.ravel(), cv=loocv)
predict_loocv4 = model_selection.cross_val_predict(classifier4, X_orig4, y_orig4.ravel(), cv=loocv)

print(results_loocv4.mean())
#print(predict_loocv4.ravel(), y_orig4.ravel())


all_predict = np.stack((predict_loocv, predict_loocv2, predict_loocv3, predict_loocv4), 1)
all_y = np.stack((y_orig.ravel(), y_orig2.ravel(), y_orig3.ravel(), y_orig4.ravel()), 1)
print(all_predict)
print("break")
print(all_y)

#Multilabel accuracy
print(np.sum((all_predict == all_y).all(1))/49)

#Count number of false alarms
print(np.greater(all_predict, all_y))

#Count number of misses
print(np.greater(all_y, all_predict))

#Count number of hits
part1 = all_y[all_predict==1]
part2 = part1[part1 == 1]
result = part2.sum()
print(result)


