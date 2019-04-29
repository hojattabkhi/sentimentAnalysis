#-*- coding: utf-8 -*-
from sklearn import *
import _pickle as cPickle
import datetime

def makeReady(model_path, vocabulary_path):
    with open(model_path, 'rb') as fid:
        clf = cPickle.load(fid)
        vectorizer = cPickle.load(open(vocabulary_path, "rb"))
    return clf, vectorizer

def predict(clf, vectorizer, sentence):
    test_data_features = vectorizer.transform([sentence])
    if not test_data_features.getnnz():
        print(datetime.datetime.now(), "\n\t", sentence, file=open("execution_logs/notValid_predicted.txt", "a"))
        return {'status': 'neural'}
    y_predicted = clf.predict(test_data_features)
    if y_predicted[0] == 1:
        print(datetime.datetime.now(),"\n\t",sentence, file=open("execution_logs/positive_predicted.txt", "a"))
        return {'status': 'positive'}
    else:
        print(datetime.datetime.now(),"\n\t",sentence, file=open("execution_logs/negative_predicted.txt", "a"))
        return {'status': 'negative'}


if __name__ == "__main__":
    # model_path = './models/KNeighbors_classifier_for_predict_sample.pk'
    # model_path = './models/Naive_bayes_classifier_for_predict_sample.pk'
    model_path = './models/svm.LinearSVC_classifier_for_predict_sample.pk'
    vocabulary_path = "vector.pickel"
    clf, vectorizer = makeReady(model_path, vocabulary_path)
    print(predict(clf, vectorizer, '﻿سلام خدمت دوستانی که قصد خرید این تبلت رو دارن باید بگم که من این محصول رو از دیجیکالا خریداری کردم اما بسیار ناراضی هستم واقعا کیفیت این تبلت پایینه اصلا پیشنهاد خریدشو بهتون نمیدم بعد از تقریبا دو هفته استفاده دیدم روشن نمیشه رفتم گفتن برد سوخته الان دقیقا تبدیل شده به یه سنگ واسه من گفتم پولی که میخوام بدم درست کنم رو نگه میدارم یه مقدار میزارم روش میرم میگیرم که خیلی سر تره پیشنهاد من درکل به شما اینه که سمت این تبلت نرید من هم گفتم قیمتش پایینه وسوسه شدم خریدم ولی الان واقعا از خریدم پشیمونم ولی از دیجیکالا به خاطر پاسگویی خوب به مشتریان تشکر میکنم یاعلی . '))
    print(predict(clf, vectorizer, '﻿قابلیت انعطاف پایینی داشت و برای کارهای کوچک اصلا مناسب نیست . '))
    print(predict(clf, vectorizer, '﻿از فرصت استفاده کنین بخرید . '))
