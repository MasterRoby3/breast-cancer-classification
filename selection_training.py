import numpy as np
import cv2 
from matplotlib import pyplot as plt
import pandas as pd # for reading and writing tables
import ntpath
from numpy import where
from collections import Counter

# Import utility functions
from util import *

# For cross validation
from sklearn.model_selection import KFold

# Other ml utils
from sklearn.metrics import confusion_matrix, classification_report
import imblearn
from imblearn.over_sampling import SMOTE

# Plot Setup Code
# Setup the defaults to make the plots look a bit nicer for the notebook
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})

import sys
classifier_type = sys.argv[1]
print("You choose the classifier:", classifier_type)
resnet_extr = sys.argv[2]
print("ResNet50 extraction:", resnet_extr)


# Read saved dataset of extracted features
dataset = pd.read_pickle("image_dataset.pkl")
labels = np.load("labels.npy")

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Method to extract features with ResNet
def neural_extr():
    deep_folder = "../deeplearning/training_set/"
    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        deep_folder,
        target_size=(224, 224),
        class_mode='binary',
        subset='training'
    )

    # Obtain class indices
    class_indices = train_generator.class_indices
    # Create a list of class names
    class_names = list(class_indices.keys())

    # Obtain labels for the training and validation sets
    train_labels = []
    for i in range(train_generator.samples):
        label = train_generator.labels[i]
        train_labels.append(label)
    np_train_labels = np.array(train_labels)

    # Load ResNet50 model
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    deep_features = model.predict_generator(train_generator)
    return deep_features, np_train_labels

# If neural network extraction is selected, this function calls the function for the extraction of features 
# and merge everything in the main dataset
if resnet_extr == "on":
    deep_feat, deep_lbl = neural_extr()
    dataset = dataset.join(pd.DataFrame(deep_feat))
    dataset.columns = dataset.columns.astype(str)

# Feature Selection
def compute_fdr(input_data, input_labels, k=5):
    index = input_labels == 0
    data_0 = input_data[~index,:].astype('float')
    data_1 = input_data[index,:].astype('float')
    m_0 = np.mean(data_0, axis = 0)
    m_1 = np.mean(data_1, axis = 0)
    std_0 = (np.std(data_0, axis = 0))**2
    std_1 = (np.std(data_1, axis = 0))**2
    fdr = ((m_0 - m_1)**2) / (std_0 + std_1)

    # Ranking
    selection_index = fdr.argsort()[::-1]
    sorted_fdr = fdr[selection_index]

    ranked_features = input_data[:,selection_index]
    selected_features = ranked_features[:,:k]
    np.array(selected_features.tolist())

    return fdr, sorted_fdr, selected_features, selection_index


# create the range 1 to 100 (data samples)
rn = range(1,len(dataset))

# KFold function returns a location (index) of the train and test samples
kf = KFold(n_splits=10, shuffle=True)


#features_to_select = 78
#features_to_select = 41
features_to_select = 200

from sklearn import preprocessing

def z_score_norm(data):
    return (data.astype(float) - np.mean(data.astype(float), axis=0)) / np.std(data.astype(float), axis=0)

def perform_kfold(kf, rn, dataset, labels, classifier_type="tree"):
    trained_classifier = []
    classes = []
    all_testlabels = []
    all_selection_train_index = []
    all_classifiers = []
    i = 0
    for train_index, test_index in kf.split(rn):
        train_features = dataset.iloc[ train_index ]
        test_features = dataset.iloc[ test_index ]

        train_labels = labels[train_index]
        test_labels = labels[test_index]
                
        # Synthetic oversampling with SMOTE
        oversample = SMOTE()
        os_train_features, os_train_labels = oversample.fit_resample(train_features, train_labels)

        # Feature selection (optional)
        pd_train_features = os_train_features
        np_train_features = os_train_features.to_numpy()
        np_train_labels = np.array(os_train_labels)
        fdr_, sorted_fdr_, selected_train_features, selection_train_index = compute_fdr(np_train_features, np_train_labels, features_to_select)
        all_selection_train_index.append(selection_train_index)

        # Replicate feature selection on testing set        
        np_test_features = test_features.to_numpy()
        ranked_test_features = np_test_features[:,selection_train_index]
        selected_test_features = ranked_test_features[:,:features_to_select]

        # TRAINING
        from sklearn import tree
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier as forest
        from sklearn.ensemble import AdaBoostClassifier as AdaBoost
        from sklearn.ensemble import VotingClassifier
        from grid_search_utils import plot_grid_search, table_grid_search
        from sklearn.model_selection import GridSearchCV


        match classifier_type:
            case "tree":
                classifier = tree.DecisionTreeClassifier(criterion = "gini", class_weight="balanced", ccp_alpha = 0.01)

                # The next commented part was used once to determine best hyperparameters
                #ab_clf = tree.DecisionTreeClassifier(random_state=42)
                #parameters = {
                #    'criterion': ["gini", "entropy", "log_loss"],
                #    'splitter': ["best", "random"],
                #    'min_weight_fraction_leaf': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
                #    'class_weight': ["balanced", None],
                #    'ccp_alpha': [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]
                #}
                #clf = GridSearchCV(ab_clf, parameters, cv=5, verbose=1, n_jobs=12)
                #clf.fit(selected_train_features, np_train_labels)
                #plot_grid_search(clf)
                #table_grid_search(clf)
            case "svm":
                classifier = SVC(C=10., gamma="scale", kernel="rbf", class_weight="balanced", cache_size=2000, probability=True, break_ties=True)
                
                selected_train_features = z_score_norm(selected_train_features)
                selected_test_features = z_score_norm(selected_test_features)

                # The next commented part was used once to determine best hyperparameters
                #ab_clf = SVC(cache_size=2000)
                #parameters = {
                #    'C': [0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.],
                #    'gamma': ["scale", "auto"],
                #    'kernel': ["rbf", "linear"],
                #    'tol': [0.005, 0.001],
                #    'class_weight': [None, "balanced"],
                #    
                #}
                #clf = GridSearchCV(ab_clf, parameters, cv=5, verbose=1, n_jobs=12)
                #clf.fit(selected_train_features, np_train_labels)
                #plot_grid_search(clf)
                #table_grid_search(clf)
                #classifier = forest(n_estimators=100, n_jobs=12, class_weight="balanced", verbose=0)

            case "forest":
                classifier = forest(criterion="gini", n_estimators=100, n_jobs=12, class_weight="balanced", verbose=0, ccp_alpha=0.00)

                # The next commented part was used once to determine best hyperparameters
                #ab_clf = forest(random_state=42)
                #parameters = {
                #    'criterion': ["gini", "entropy", "log_loss"],
                #    'n_estimators': [100, 200, 300, 500, 1000],
                #    'class_weight': ["balanced", "balanced_subsample"],
                #    'ccp_alpha': [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]
                #}
                #clf = GridSearchCV(ab_clf, parameters, cv=5, verbose=1, n_jobs=12)
                #clf.fit(selected_train_features, np_train_labels)
                #plot_grid_search(clf)
                #table_grid_search(clf)
                #classifier = forest(n_estimators=100, n_jobs=12, class_weight="balanced", verbose=0)
            case "ada":
                classifier = AdaBoost(n_estimators=200, learning_rate=0.99)

                # The next commented part was used once to determine best hyperparameters
                #ab_clf = AdaBoost(random_state=42)
                #parameters = {
                #    'n_estimators': [200, 300, 500, 1000],
                #    'learning_rate': [(0.97 + x / 100) for x in range(0, 8)],
                #    'algorithm': ['SAMME', 'SAMME.R']
                #}
                #clf = GridSearchCV(ab_clf, parameters, cv=5, verbose=1, n_jobs=12)
                #clf.fit(selected_train_features, np_train_labels)
                #plot_grid_search(clf)
                #table_grid_search(clf)
            case "voting":
                ada = AdaBoost(n_estimators=200, learning_rate=0.99)
                tree = tree.DecisionTreeClassifier(criterion = "gini", class_weight="balanced", ccp_alpha = 0.01)
                classifier = VotingClassifier(estimators=[('ada', ada),('tree',tree)],voting='soft')
                
            case _:
                classifier = tree.DecisionTreeClassifier(criterion = "gini")

        classifier = classifier.fit(selected_train_features, np_train_labels)
        all_classifiers.append(classifier)

        # Predict on TESTING samples
        classes = np.concatenate((classes, classifier.predict(selected_test_features)), axis=0)
        all_testlabels = np.concatenate((all_testlabels, test_labels), axis = 0)

        i+=1

    return classes, all_testlabels, all_selection_train_index, all_classifiers


# Perform cross validation
testsample_predictedclasses, testsample_labels, selection_train_index, all_classifiers = perform_kfold(kf, rn, dataset, labels, classifier_type=classifier_type)


# Create heatmap from the confusion matrix
def createConfMatrix(class_names, matrix):
    class_names=[0, 1] 
    tick_marks = [0.5, 1.5]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="Blues", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label'); plt.xlabel('Predicted label')
    plt.yticks(tick_marks, class_names); plt.xticks(tick_marks, class_names)
    plt.show()
    
# Create a confusion matrix
cnf_matrix = confusion_matrix(testsample_labels, testsample_predictedclasses)
createConfMatrix(matrix=cnf_matrix, class_names=[0, 1])

def print_performance(cnf_matrix):
    TP = cnf_matrix[1][1]
    TN = cnf_matrix[0][0]
    FP = cnf_matrix[0][1]
    FN = cnf_matrix[1][0]
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate mis-classification
    conf_misclassification = 1- conf_accuracy
    
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))

    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    print('Acc:', conf_accuracy)
    print('Err:', conf_misclassification)    
    print('Sen:', conf_sensitivity)
    print('Spe:', conf_specificity)

print_performance(cnf_matrix)

# Method for extraction with ResNet for external testing
def neural_extr_test():
    deep_folder = "../deeplearning/"
    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(
        deep_folder,
        target_size=(224, 224),
        classes = ['testing_set']
    )

    # Load ResNet50 model
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    deep_features = model.predict(test_generator)
    return deep_features


# Save data external testing
this_classifier = all_classifiers

exttest = pd.read_pickle("test.pkl")
if resnet_extr == "on":
    deep_feat = neural_extr_test()
    exttest = exttest.join(pd.DataFrame(deep_feat))
    exttest.columns = exttest.columns.astype(str)


np_extst_features = exttest.to_numpy()
if classifier_type == "svm":
    np_extst_features = z_score_norm(np_extst_features)

# Use the classifier on the test data
tst_class = [0] * 100
tst_probabilities = [0] * 100

for i in range(0,10):
  ranked_extst_features = np_extst_features[:,selection_train_index[i]]
  selected_extst_features = ranked_extst_features[:,:features_to_select]
  selected_extst_features = np.array(selected_extst_features.tolist())
  
  tst_class = tst_class + ((this_classifier[i].predict(selected_extst_features))/10)
  prob = this_classifier[i].predict_proba(selected_extst_features)
  tst_probabilities = tst_probabilities + ((prob[:,1])/10)

  i+=1


sample_submission = pd.read_csv("../submissions/sample_submission_auc.csv")
submission = pd.DataFrame()
submission['Id'] = sample_submission['Id']
submission['Predicted'] = tst_probabilities
#submission['Predicted'] = tst_class
submission.to_csv('../submissions/submission.csv', index=False)