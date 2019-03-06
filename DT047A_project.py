import csv
import graphviz 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import feature_selection
from sklearn import naive_bayes
from sklearn import tree
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


data = []
with open('adult_preprocess_test.csv') as csv_file:
    file = csv.reader(csv_file, delimiter=',')
    badlines = 0
    for line in file:
        if '?' in line or ' ?' in line or ' ? ' in line or len(line) == 0 or '? ' in line:
            badlines += 1
        else:
            for elem in range(len(line)):
                line[elem] = line[elem].replace(" ","")
                line[elem] = str(line[elem])
            data.append(line)

print(len(data))
print('num badlines = ', badlines)
print(data[0])

# now data is clean. Got 30162 good instances

data_attributes = ['age', 'workclass', 'fnlwgt', 'education' ,'education_num', 'marital_status',
                    'occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week'
                    , 'native_country','Target_class']
data_attributes_trimmed = ['age', 'workclass', 'fnlwgt', 'education' ,'education_num', 'marital_status',
                             'occupation','relationship','race','sex','hours_per_week'
                            , 'native_country','Target_class']

data_dictionary = {
                   'age': ['15-24','25-34','35-44','45-54','55-64','65-74','75-84','85-94'],
                    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc','Federal-gov', 'Local-gov','State-gov', 'Without-pay', 'Never-worked'],
                    'fnlwgt': ['13769-160861', '160862-307955', '307956-455048', '455049-602142', '602143-749236', '896330-1043423', '749237-896329', '1043424-1190516', '1190517-1337610', '1337611-1484704'],
                    'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
                    'education_num': 'continuous',
                    'marital_status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
                    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
                    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
                    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
                    'sex': ['Female', 'Male'],
                    'capital-gain': 'continuous',# not in preproccessed version
                    'capital-loss': 'continuous',#not in preproccessed version
                    'hours_per_week': 'continuous',
                    'native_country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal','Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
                    'Target_class': ['>50K', '<=50K']
}
targets = []
for row in data:
    targets.append(row[12])
def cleandata(arr, attribs, meta):
	for i in range(len(arr)):
		for j in range(len(arr[0])):
			if meta[attribs[j]] is 'continuous':
				arr[i][j] = int(arr[i][j])
			else:
				arr[i][j] = meta[attribs[j]].index(arr[i][j])


cleandata(data, data_attributes_trimmed, data_dictionary)


chidata = np.array(data)


chi2_arr, pval = feature_selection.chi2(chidata[:,[0,1,2,3,4,5,6,7,8,9,10,11]].astype(int),chidata[:,12])
'''
def chi2_correlation_test(data_array, target, target_name):
    chi2_arr, pval = feature_selection.chi2(data_array.astype(int),target) #spits out the amount of error to independence. High value == Corrolated, small == independent.
    f = open("out.txt", "a")
    f.write(target_name + 'CHI2 data:  \n' + str(chi2_arr)  + '\n\n')
    f.write(target_name + 'PVAL data   \n' + str(pval) + '\n\n\n')
    
chi2_correlation_test(chidata[:,[1,2,3,4,5,6,7,8,9,10,11]].astype(int),chidata[:,0], data_attributes[0])
chi2_correlation_test(chidata[:,[0,2,3,4,5,6,7,8,9,10,11]].astype(int),chidata[:,1], data_attributes[1])
chi2_correlation_test(chidata[:,[0,1,3,4,5,6,7,8,9,10,11]].astype(int),chidata[:,2], data_attributes[2])
chi2_correlation_test(chidata[:,[0,1,2,4,5,6,7,8,9,10,11]].astype(int),chidata[:,3], data_attributes[3])
chi2_correlation_test(chidata[:,[0,1,2,3,5,6,7,8,9,10,11]].astype(int),chidata[:,4], data_attributes[4])
chi2_correlation_test(chidata[:,[0,1,2,3,4,6,7,8,9,10,11]].astype(int),chidata[:,5], data_attributes[5])
chi2_correlation_test(chidata[:,[0,1,2,3,4,5,7,8,9,10,11]].astype(int),chidata[:,6], data_attributes[6])
chi2_correlation_test(chidata[:,[0,1,2,3,4,5,6,8,9,10,11]].astype(int),chidata[:,7], data_attributes[7])
chi2_correlation_test(chidata[:,[0,1,2,3,4,5,6,7,9,10,11]].astype(int),chidata[:,8], data_attributes[8])
chi2_correlation_test(chidata[:,[0,1,2,3,4,5,6,7,8,10,11]].astype(int),chidata[:,9], data_attributes[9])
chi2_correlation_test(chidata[:,[0,1,2,3,4,5,6,7,8,9,11]].astype(int),chidata[:,10], data_attributes[10])
chi2_correlation_test(chidata[:,[0,1,2,3,4,5,6,7,8,9,10]].astype(int),chidata[:,11], data_attributes[11])

Y = {'age':chidata[0], 'workclass':chidata[1], 'fnlwgt':chidata[2], 'education':chidata[3] ,'education_num':chidata[4], 'marital_status':chidata[5],
                    'occupation':chidata[6],'relationship':chidata[7],'race':chidata[8],'sex':chidata[9],'hours_per_week':chidata[10]
                    , 'native_country':chidata[11],'Target_class':chidata[12]}

'''
chi2_selector = feature_selection.SelectKBest(feature_selection.chi2, k=5)
chi2_selector.fit(chidata[:,[0,1,2,3,4,5,6,7,8,9,10,11]].astype(int),chidata[:,12])

reduced = chi2_selector.get_support(indices=True)
print(reduced)
slim_data = np.append(reduced,12)
print(slim_data)
transformed_data = chidata[:,slim_data]
legend = np.array(data_attributes_trimmed)
print(legend[slim_data])
tree_legend = legend[slim_data]
print(transformed_data)

#print('chi2 res: ', chi2_arr)
#print()
#print('pval; ',pval)


clf = tree.DecisionTreeClassifier(max_depth=4,min_samples_leaf=600,min_samples_split=600)
clf = clf.fit(transformed_data[:,[0,1,2,3,4]],transformed_data[:,5] )
 
tree_data_export = tree.export_graphviz(clf, out_file=None,  feature_names=legend[reduced],  class_names=['More than 50k','Less than or Equal to 50K'],  filled=True, rounded=True,  special_characters=True)  
graph = graphviz.Source(tree_data_export)  
graph.render("IncomeTree")

data = transformed_data.tolist()
testsize = int(len(data) / 10)
testindex = 0
accuracy = []
tprs_clf = []
base_fpr_clf = np.linspace(0, 1, 101)
for i in range(10):

    test = data[testindex : testindex + testsize]
    train = data[0:testindex] + data[testindex + testsize : len(data)]
    testindex += testsize
    train = np.array(train)
    test = np.array(test)
    clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=600, min_samples_split=600)
    clf.fit(train[:,:-1], train[:,-1])
    ptest = clf.predict(test[:,:-1])

    correct = 0
    for i in range(len(ptest)):
        if ptest[i] == test[i][-1]:
            correct += 1

    y_test_predictions = clf.predict_proba(test[:,:-1]) 
    fpr, tpr, _ = roc_curve(test[:,-1],y_test_predictions[:, 1])

    plt.plot(fpr, tpr, 'g', alpha=0.15)
    tpr = interp(base_fpr_clf, fpr, tpr)
    tpr[0] = 0.0
    tprs_clf.append(tpr)
    acc = float(correct) / len(ptest)
    accuracy.append(acc)
    print(acc)
    print ('Confusion matrix:')
    print (sk_confusion_matrix(test[:,-1], ptest))


print()
print('avg: ', sum(accuracy) / len(accuracy))
print('-----------------------------------------')
print()
tprs_clf = np.array(tprs_clf)
mean_tprs_clf = tprs_clf.mean(axis=0)
std = tprs_clf.std(axis=0)

tprs_clf_upper = np.minimum(mean_tprs_clf + std, 1)
tprs_clf_lower = mean_tprs_clf - std


plt.plot(base_fpr_clf, mean_tprs_clf, 'g')
plt.fill_between(base_fpr_clf, tprs_clf_lower, tprs_clf_upper, color='grey', alpha=0.3)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal', 'datalim')
plt.show()
#----------------------------------------------------------------------------------------------

gnb = naive_bayes.GaussianNB()
accuracy_array = []
cut_point = int(len(transformed_data))*0.1
cutpoint = int(cut_point)

X = transformed_data[:,[0,1,2,3,4]]
Y = transformed_data[:,5]
x = X.tolist()
y= Y.tolist()

tprs_nb = []
base_fpr_nb = np.linspace(0, 1, 101)
for i in range(10):

    start = int(cutpoint * i)
    stop = int(cutpoint * (i+1))
    
    X_train = x[0:start]+x[stop:len(x)]
    X_test = x[start:stop]
    y_train = y[0:start]+y[stop:len(x)]
    y_test = y[start:stop]

    X_train = np.array(X_train)
    X_test = np.array( X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    #print(len(X_train))

    gnb.fit(X_train, y_train) 

    y_train_predictions = gnb.predict(X_train) 

    y_test_predictions = gnb.predict(X_test) 

    y_score = gnb.fit(X_train, y_train).predict_proba(X_test)
    acc = (y_test_predictions == y_test).sum().astype(float)/(y_test.shape[0])
    print ('Accuracy:',acc)
    accuracy_array.append(acc)
    #print ('Classification report:')
    #print (classification_report(y_test, y_test_predictions))

    print ('Confusion matrix:')
    print (sk_confusion_matrix(y_test, y_test_predictions))

    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])

    plt.plot(fpr, tpr, 'b', alpha=0.15)
    tpr = interp(base_fpr_nb, fpr, tpr)
    tpr[0] = 0.0
    tprs_nb.append(tpr)

tprs_nb = np.array(tprs_nb)
mean_tprs_nb = tprs_nb.mean(axis=0)
std = tprs_nb.std(axis=0)

tprs_nb_upper = np.minimum(mean_tprs_nb + std, 1)
tprs_nb_lower = mean_tprs_nb - std


plt.plot(base_fpr_nb, mean_tprs_nb, 'b')
plt.fill_between(base_fpr_nb, tprs_nb_lower, tprs_nb_upper, color='grey', alpha=0.3)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal', 'datalim')
plt.show()
    
print()
print('AVG accuracy from Cross-Validation, Naive Bayers:',float(sum((accuracy_array)))/len(accuracy_array))
print('-------------------------------------------------------')

#-------------------------------------------------------------------------------------------------


#clf_rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
#clf_rf = clf_rf.fit(transformed_data[:,[0,1,2,3,4]],transformed_data[:,5] )
 
#tree_data_export = tree.export_graphviz(clf_rf, out_file=None,  feature_names=legend[reduced],  class_names=['More than 50k','Less than or Equal to 50K'],  filled=True, rounded=True,  special_characters=True)  
#graph = graphviz.Source(tree_data_export)  
#graph.render("IncomeTree_rf")

data = transformed_data.tolist()
testsize = int(len(data) / 10)
testindex = 0
accuracy = []
tprs_clf_rf = []
base_fpr_clf_rf = np.linspace(0, 1, 101)
for i in range(10):

    test = data[testindex : testindex + testsize]
    train = data[0:testindex] + data[testindex + testsize : len(data)]
    testindex += testsize
    train = np.array(train)
    test = np.array(test)
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=600, min_samples_split=600,random_state=0)
    clf_rf.fit(train[:,:-1], train[:,-1])
    ptest = clf_rf.predict(test[:,:-1])

    correct = 0
    for i in range(len(ptest)):
        if ptest[i] == test[i][-1]:
            correct += 1

    y_test_predictions = clf_rf.predict_proba(test[:,:-1]) 
    fpr, tpr, _ = roc_curve(test[:,-1],y_test_predictions[:, 1])

    plt.plot(fpr, tpr, 'r', alpha=0.15)
    tpr = interp(base_fpr_clf, fpr, tpr)
    tpr[0] = 0.0
    tprs_clf_rf.append(tpr)
    acc = float(correct) / len(ptest)
    accuracy.append(acc)
    print(acc)
    print ('Confusion matrix:')
    print (sk_confusion_matrix(test[:,-1], ptest))


print()
print('avg: ', sum(accuracy) / len(accuracy))
print('-----------------------------------------')
print()
tprs_clf_rf = np.array(tprs_clf_rf)
mean_tprs_clf_rf = tprs_clf_rf.mean(axis=0)
std = tprs_clf_rf.std(axis=0)

tprs_clf_rf_upper = np.minimum(mean_tprs_clf_rf + std, 1)
tprs_clf_rf_lower = mean_tprs_clf_rf - std


plt.plot(base_fpr_clf_rf, mean_tprs_clf_rf, 'r')
plt.fill_between(base_fpr_clf_rf, tprs_clf_rf_lower, tprs_clf_rf_upper, color='grey', alpha=0.3)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal', 'datalim')
plt.show()