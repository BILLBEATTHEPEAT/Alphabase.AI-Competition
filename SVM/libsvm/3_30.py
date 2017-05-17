#!/usr/bin/env python


import pandas as pd
import numpy as np
from svmutil import *


def main():

    print("Loading the data ...")
    # Load the data from the npy files
    # X_train = np.load('X_train.npy').tolist()
    # X_test = np.load('X_test.npy').tolist()
    X_train = np.load('New_train.npy').tolist()
    X_test = np.load('New_test.npy').tolist()
    label = np.load('Label.npy').tolist()
    test = pd.read_csv('test.csv', header=0)
    ID = test['ID']
    # print train.shape, '\n', test.shape, label.shape, ID.shape

    ###################
    ###################
    # Create a SVM Model.
    print("Training the SVM model ...")
    # prob_train = problem(label, X_train)
    prob_train = svm_problem(label, X_train)
    # param = parameter('-s 2 -C')
    param = svm_parameter('-t 2 -c 8 -h 0 -b 1 -v 5')
    # best_C, best_rate = train(prob_train, param)
    # param_ter = '-q -s 0 -c %s' % best_C
    # m = train(label, X_train, param_ter)
    m = svm_train(prob_train, param)

    ##
    #################################
    ##################################
    # Step 3. Predict the Competition Data with the newly trained model
    print("Predicting the Competition Data...")
    label_test = np.zeros(len(X_test)).tolist()
    # p_labels, p_acc, p_vals = predict(label_test, X_test, m, '-b 1')
    p_label, p_acc, p_vals = svm_predict(label_test, X_test, m, '-b 1')

    print '------------------------'
    print 'Predict Complete'
    y_test = p_vals
    # Get the probabilty of being 1.
    pred_original = np.array(y_test)
    print pred_original.shape
    np.save('result', p_vals)
    # del X_train

    result = np.load('result.npy')
    pred_original = result[:, 1]
    # ID = np.load('ID.npy').tolist()

    pred = pred_original.reshape(len(pred_original), 1)
    pred_df = pd.DataFrame(data={'Target': pred[:, 0]})
    submissions = pd.DataFrame(ID).join(pred_df)

    ##
    #################################
    ##################################
    # Step 4. Write the CSV File and Get Ready for Submission
    # Save the predictions out to a CSV file
    print("Writing predictions to abai_submissions.csv")
    submissions.to_csv("submissions_3.csv", index=False)

    ##
    #################################


if __name__ == '__main__':
    main()
