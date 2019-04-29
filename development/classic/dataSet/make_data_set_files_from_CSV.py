#-*- coding: utf-8 -*-

import pandas as pd
import os

train_set_folder_name = "sentiment_data_set"
csv_file = "CleanedReviews.csv"

positive_folder_name = train_set_folder_name + "/positive"
negative_folder_name = train_set_folder_name + "/negative"
neural_folder_name = train_set_folder_name + "/neural"
if not os.path.exists(train_set_folder_name):
    os.makedirs(train_set_folder_name)
    os.makedirs(positive_folder_name)
    os.makedirs(negative_folder_name)
    # os.makedirs(neural_folder_name)

    batch_file = pd.read_csv(csv_file, encoding='utf-8')
    lines = batch_file.values

    words = []

    positive_counter = 0
    negative_counter = 0
    neural_counter = 0
    for line in batch_file.values:
        try:
            if line[0] > 3 and positive_counter < 8878:
                positive_counter += 1
                file = open(positive_folder_name + "/" + str(positive_counter), 'wb')
            elif line[0] < 3 :
                negative_counter += 1
                file = open(negative_folder_name + "/" + str(positive_counter), 'wb')
            elif line[0] == 3:
                neural_counter += 1
                # file = open(neural_folder_name + "/" + str(positive_counter), 'wb')

            file.write(line[1].encode("utf-8"))
            file.close()
        except Exception as e:
            print(e)
    print('positive files:\t', positive_counter)
    print('negative files:\t', negative_counter)
    print('neural files:\t', neural_counter)