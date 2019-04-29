#-*- coding: utf-8 -*-
from sklearn import *
import _pickle as cPickle
import os


def get_classifier(method):
    if method == "KNeighbors":
        n_neighbors = 11
        weights = 'distance'
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    elif method == "Naive_bayes":
        clf = naive_bayes.MultinomialNB()
    elif method == "svm.LinearSVC":
        clf = svm.LinearSVC()
    return clf

def classify(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf

def test_classifier_and_report(X_test, y_test, clf, main_data, test_index, report=False):
    y_predicted = clf.predict(X_test)
    precision = metrics.precision_score(y_test, y_predicted, main_data.target_names)
    recall = metrics.recall_score(y_test, y_predicted,main_data.target_names)
    F1_score = metrics.f1_score(y_test, y_predicted,main_data.target_names)
    accuracy = metrics.accuracy_score(y_test, y_predicted,main_data.target_names)
    print("--------------- REPORT ---------------")
    print('\tPrecision_score:\t', precision)
    print('\tRecall:\t', recall)
    print('\tF1_score:\t', F1_score)
    print('\tAccuracy:\t', accuracy)
    print('\tConfusion Matrix:')
    print(metrics.confusion_matrix(y_test, y_predicted), "\n")
    if report:
        for i in range(0, len(test_index)):
            print (
                main_data.data[test_index[i]].decode('utf8'),
                "\n\treal:", main_data.target_names[y_test[i]],
                "\tpredict:", main_data.target_names[y_predicted[i]], '\n',
                file=open("detailed_predictions.txt", "a")
            )
    return precision, recall, F1_score, accuracy

if __name__ == '__main__':
    path = "./dataSet/sentiment_data_set"
    methods = ["svm.LinearSVC", "Naive_bayes", "KNeighbors"]

    count_vector = feature_extraction.text.CountVectorizer()
    tf_transformer = feature_extraction.text.TfidfTransformer(use_idf=True)
    kFold = model_selection.KFold(n_splits=10)

    # load data
    print('Loading files into memory')
    files = datasets.load_files(path)
    print('Fit and tranform')
    print('Calculating BOW')
    word_counts = count_vector.fit_transform(files.data)

    cPickle.dump(count_vector, open("vector.pickel", "wb"))

    print('Calculating TFIDF')
    data = tf_transformer.fit_transform(word_counts)

    # Split in train/test
    for classify_method in methods:
        clf = get_classifier(classify_method)
        print("\n...................... ", classify_method, " Classifier ..........................")
        if not os.path.exists("models"):
            os.makedirs("models")

        # generate model with all dataset for classify samples out of dataSet
        clf = classify(clf, data, files.target)
        model_path = os.path.join("models", classify_method + "_classifier_for_predict_sample.pk")
        with open(model_path, 'wb') as fid:
            cPickle.dump(clf, fid)

        precision = []
        recall = []
        F1_score = []
        accuracy = []
        fold_number = 1
        for train_index, test_index in kFold.split(files.data):
            X_train = data[train_index]
            y_train = files.target[train_index]
            X_test = data[test_index]
            y_test = files.target[test_index]

            clf = classify(clf, X_train, y_train)
            # save the classifier
            model_path = os.path.join("models", classify_method + "_" + str(fold_number) + "_classifier.pk")
            with open(model_path, 'wb') as fid:
                cPickle.dump(clf, fid)
            print("Fold: \t" + str(fold_number))
            fold_precision, fold_recall, fold_F1_score, fold_accuracy = test_classifier_and_report(
                X_test, y_test, clf, files, test_index, True
            )
            precision.append(fold_precision)
            recall.append(fold_recall)
            F1_score.append(fold_F1_score)
            accuracy.append(fold_accuracy)
            fold_number += 1
        print("Mean of ", fold_number - 1 , "fold precision:\t", sum(precision) / len(precision))
        print("Mean of ", fold_number - 1, "fold recall:\t", sum(recall) / len(recall))
        print("Mean of ", fold_number - 1, "fold F1_score:\t", sum(F1_score) / len(F1_score))
        print("Mean of ", fold_number - 1, "accuracy:\t", sum(accuracy) / len(accuracy))
        print("#########################################################################\n")
