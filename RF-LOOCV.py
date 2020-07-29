import numpy as np
import pandas
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from math import sqrt
from numpy.random import seed
from keras.regularizers import l1
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
import random
from skmultilearn.problem_transform import BinaryRelevance
from sklearn import metrics




dataframe = pandas.read_csv("FullTest.csv", header = 0)




dataset = dataframe.values
# split into input (X) and output (Y) variables
X_orig = dataset[:,0:8].astype(float)
y_orig = dataset[:,8:12].astype(float)


        

################################################################################################################
search_list3 = [2, 3, 4, 5, 6, 7, 8]
search_list1 = [2, 4]
search_list2 = [10, 20, 30, 50, 100, 200]

results = np.empty((0, 4), int)

for m in search_list3:
    for l in search_list2:
        for t in search_list1:
            classifier = BinaryRelevance(
            classifier = RandomForestClassifier(bootstrap = True, n_estimators = l, criterion = 'gini', random_state = 6,
            max_depth = 20, max_features = m, max_leaf_nodes = None, 
            min_impurity_decrease = 0.0, min_impurity_split = None, min_samples_split = 2 ,
            min_samples_leaf = t, min_weight_fraction_leaf = 0.0, n_jobs = -1, oob_score = False,
            verbose = 0, warm_start = False), require_dense = [False, True])


            i = 0
            j = 0
            for p in range(0, 49):
                X_copy = X_orig[(p):(p+1)]  #Slice the ith element from the numpy array
                y_copy = y_orig[(p):(p+1)]
                X_model = X_orig
                y_model = y_orig  #Set X and y equal to samples and labels


                X_model = np.delete(X_model, p, axis = 0)  #Create a new array to train the model with slicing out the ith item for LOOCV
                y_model = np.delete(y_model, p, axis = 0)

                train_set = np.concatenate((X_model, y_model), axis = 1) #combine numpy matrices 


                classifier.fit(X_model, y_model)
                prediction = classifier.predict(X_copy)
                #print(prediction.toarray(), y_copy)
                results = np.append(results, np.array(prediction.toarray()), axis = 0)
                if np.array_equal(y_copy, prediction.toarray()):
                    j = j + 1
				    #print(y_copy, prediction)
                if np.not_equal:
				    #print(y_copy, prediction)
                    pass
            print(j/49)

att = results[:,0]
esc = results[:,1:2]
ns = results[:,2:3]
tang = results[:,3:4]

y_att = y_orig[:,0]
y_esc = y_orig[:,1:2]
y_ns = y_orig[:,2:3]
y_tang = y_orig[:,3:4]


att_f1 = metrics.f1_score(y_att, att)
print("The results of F1 score are :" + str(att_f1))

esc_f1 = metrics.f1_score(y_esc, esc)
print("The results of F1 score are :" + str(esc_f1))

ns_f1 = metrics.f1_score(y_ns, ns)
print("The results of F1 score are :" + str(ns_f1))

tang_f1 = metrics.f1_score(y_tang, tang)
print("The results of F1 score are :" + str(tang_f1))

print((att_f1 + esc_f1 + ns_f1 + tang_f1)/4)



