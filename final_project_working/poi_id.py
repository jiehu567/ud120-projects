#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

persons = data_dict.keys()
keys = data_dict[data_dict.keys()[1]].keys()

# Check 'NaN' value of each feature in dict_data
# NA_count = {}

# for person in persons:
#     for key in keys:
#         if data_dict[person][key] == 'NaN':
#            if not NA_count.has_key(key):
#                 NA_count[key] = 1
#             else:
#                 NA_count[key] += 1
# print NA_count


# The NA count of each feature in dataset:

# {'bonus': 64,
#  'deferral_payments': 107,
#  'deferred_income': 97,
#  'director_fees': 129,
#  'email_address': 35,
#  'exercised_stock_options': 44,
#  'expenses': 51,
#  'from_messages': 60,
#  'from_poi_to_this_person': 60,
#  'from_this_person_to_poi': 60,
#  'loan_advances': 142,
#  'long_term_incentive': 80,
#  'other': 53,
#  'restricted_stock': 36,
#  'restricted_stock_deferred': 128,
#  'salary': 51,
#  'shared_receipt_with_poi': 60,
#  'to_messages': 60,
#  'total_payments': 21,
#  'total_stock_value': 20}


# Before doing any feature selection and creation work, I remove the 'NaN'
# Because for financial data, 'NaN' most likely mean he/she had no such income, so it's 0
# And for email data, it can also be 0 if there's no such count
for person in persons:
    for key in keys:
        if data_dict[person][key] == 'NaN' and key != 'email_address':
            data_dict[person][key] = 0


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Here I list all features I intuitively think relevant to my target: label poi
features_list = ['poi',
				 'deferral_payments',
				 'expenses',
                 'total_stock_value',
                 'deferred_income',
                 'total_payments',
                 'loan_advances',
                 'long_term_incentive',
                 'other'] 



### =======================================================
### Task 2: Remove outliers

## Use visualize, I find an observation 'TOTAL' is far from center, and then delete it
# features_list = ['deferral_payments','expenses']
# my_dataset = data_dict
# data = featureFormat(my_dataset, features_list, sort_keys = True)
# import matplotlib.pyplot as plt

# for point in data:
#    deferral_payments = point[0]
#    expenses = point[1]
#    plt.scatter( deferral_payments, expenses )

# plt.xlabel("deferral_payments")
# plt.ylabel("expenses")
# plt.title("Search for outliers")
# plt.show()


# for person in persons:
#    if data_dict[person]['deferral_payments'] > 20000000:
#        print person
## here will print 'TOTAL'

data_dict.pop('TOTAL')
persons.remove('TOTAL')

### =======================================================
### Task 3: Create new feature(s)

# 'fixed_income': earned from how they contribute to work
#  salary + bonus
# 'stock_income': all income from stock
# restricted_stock_deferred + exercised_stock_options + restricted_stock
# 'email_proportion_with_poi': 
# proportion of their emails frequency with poi over all email

for person in persons:
    salary = float(data_dict[person]['salary'])
    bonus = float(data_dict[person]['bonus'])
    restricted_stock_deferred = float(data_dict[person]['restricted_stock_deferred'])
    exercised_stock_options = float(data_dict[person]['exercised_stock_options'])
    restricted_stock = float(data_dict[person]['restricted_stock'])
    
    from_this_person_to_poi = float(data_dict[person]['from_this_person_to_poi'])
    shared_receipt_with_poi = float(data_dict[person]['shared_receipt_with_poi'])
    from_poi_to_this_person = float(data_dict[person]['from_poi_to_this_person'])
    to_messages = float(data_dict[person]['to_messages'])
    from_messages = float(data_dict[person]['from_messages'])
    
    data_dict[person]['fixed_income'] = salary + bonus 
    data_dict[person]['stock_income'] = (restricted_stock_deferred + \
                                         exercised_stock_options + \
                                         restricted_stock)
    data_dict[person]['email_proportion_with_poi'] = (from_this_person_to_poi + \
                                                         shared_receipt_with_poi + \
                                                         from_poi_to_this_person)/ \
                                                        (to_messages + from_messages + 1)
    # Some persons do not have any email so I add one to prevent 0 as denominator                                     
    

features_list = features_list + ['fixed_income', 'stock_income', 'email_proportion_with_poi']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Rescale data
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)

# use sklearn KBest to select the best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

data_new_best = SelectKBest(f_classif).fit(features_minmax,labels)



### Print score of each feature
# import operator

# scores = data_new_best.scores_
# score_dict = {}
# for ii in range(11):
#     score_dict[features_list[ii+1]] = round(scores[ii],2)

# sorted_score_dict = sorted(score_dict.items(), \
# 							 key=operator.itemgetter(1), \
# 							 reverse=True)
# print sorted_score_dict

## will print:
## ------- Score of features --------
## [('total_stock_value', 24.47),
##  ('fixed_income', 22.89),
##  ('stock_income', 22.78),
##  ('deferred_income', 11.6),
##  ('email_proportion_with_poi', 10.26),
##  ('long_term_incentive', 10.07),
##  ('total_payments', 8.87),
##  ('loan_advances', 7.24),
##  ('expenses', 6.23),
##  ('other', 4.2),
##  ('deferral_payments', 0.22)]

# select best 5 features, re-extract data
features_list = ['poi',
                 'total_stock_value', 
                 'fixed_income', 
                 'stock_income', 
                 'deferred_income', 
                 'email_proportion_with_poi']

my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### =======================================================
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import precision_score, recall_score
# import numpy as np

## classifier test function
# def classifer_tester(classifier, features, labels, parameters, iterations=100):
    
#     precision = []
#     recall = []
#     accuracy = []
    
#     for ii in range(iterations):
#         features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=ii)
#         grid_search = GridSearchCV(classifier, parameters)
#         grid_search.fit(features_train, labels_train)
#         predictions = grid_search.predict(features_test)
#         precision.append(precision_score(labels_test, predictions))
#         recall.append(recall_score(labels_test, predictions))
#         accuracy.append(accuracy_score(labels_test, predictions))
    
#     precision_mean = np.array(precision).mean()
#     recall_mean = np.array(recall).mean()
#     accuracy_mean = np.array(accuracy).mean()
    
#     print '------------------------'
#     print 'Accuracy: %s' % "{:,.2f}".format(round(accuracy_mean, 2)) 
#     print 'Precision: %s' % "{:,.2f}".format(round(precision_mean, 2))
#     print 'Recall   : %s' % "{:,.2f}".format(round(recall_mean, 2))
    
#     avg_F1 = 2 * (precision_mean * recall_mean) / (precision_mean + recall_mean)
#     print 'F1 score:  %s' % "{:,.2f}".format(round(avg_F1, 2))
    
#     print 'Best parameters:\n'
#     best_parameters = grid_search.best_estimator_.get_params() 
#     for parameter_name in sorted(parameters.keys()):
#         print '%s: %r' % (parameter_name, best_parameters[parameter_name])


## Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# parameters = {}
# grid_search = GridSearchCV(clf, parameters)
# print '\nGaussian Naive Bayes:'
# classifer_tester(clf, features, labels, parameters)


# Gaussian Naive Bayes:
# ------------------------
# Precision: 0.43
# Recall   : 0.37
# F1 score:  0.40
# Best parameters:


# ## Decision Tree
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# parameters = {'criterion': ['gini', 'entropy'],
#                'min_samples_split': [2, 10, 20],
#                'max_depth': [None, 2, 5, 10],
#                'min_samples_leaf': [1, 5, 10],
#                'max_leaf_nodes': [None, 5, 10, 20]}

# grid_search = GridSearchCV(clf, parameters)
# print '\nDecision Tree:'
# classifer_tester(clf, features, labels, parameters)


# Decision Tree:
# ------------------------
# Precision: 0.16
# Recall   : 0.14
# F1 score:  0.15
# Best parameters:

# criterion: 'gini'
# max_depth: None
# max_leaf_nodes: None
# min_samples_leaf: 10
# min_samples_split: 2


# ## Random Forest
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()
# parameters = {'criterion': ['gini', 'entropy'],
#                'min_samples_split': [2, 10, 20],
#                'max_depth': [None, 2, 5, 10],
#                'min_samples_leaf': [1, 5, 10],
#                'max_leaf_nodes': [None, 5, 10, 20]}

# grid_search = GridSearchCV(clf, parameters)
# print '\nRandom Forest:'
# classifer_tester(clf, features, labels, parameters)


# Random Forest:
# ------------------------
# Precision: 0.30
# Recall   : 0.15
# F1 score:  0.20
# Best parameters:

# criterion: 'gini'
# max_depth: 2
# max_leaf_nodes: None
# min_samples_leaf: 1
# min_samples_split: 10

# # AdaBoost
# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier()
# parameters = {'n_estimators': [10, 30, 50],
#               'algorithm': ['SAMME', 'SAMME.R'],
#               'learning_rate': [.5, 1, 1.5]}
# grid_search = GridSearchCV(clf, parameters)
# print '\nAdaBoost:'
# classifer_tester(clf, features, labels, parameters)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


clf = GaussianNB()

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)