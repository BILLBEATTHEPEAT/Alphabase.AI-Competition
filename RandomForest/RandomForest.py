#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


def main():
    ##################################
    # Load alphabase.ai data
    print("Loading the data ...")
    # Load the data from the CSV files
    # X_train = np.load('X_train.npy').tolist()
    # X_test = np.load('X_test.npy').tolist()
    X_train = np.load('New_train4.npy').tolist()
    X_test = np.load('New_test4.npy').tolist()
    # X_train = X_train[:, :4].tolist()
    # X_test = X_test[:, :4].tolist()
    label = np.load('Label4.npy').ravel()
    test = pd.read_csv('test.csv', header=0)
    ID = test['ID']
    '''
    # param_test1 = {'n_estimators': range(60, 121, 10),
    #                'max_depth': range(3, 14, 2),
    #                'min_samples_split': range(50, 201, 20)}
    param_test1 = {'n_estimators': range(100, 500, 50)}
    gsearch1 = GridSearchCV(
        estimator=RandomForestClassifier(oob_score=True,
                                         min_samples_split=130,
                                         min_samples_leaf=20,
                                         random_state=10,
                                         max_depth=7,
                                         max_features='sqrt'),
        param_grid=param_test1, scoring='neg_log_loss', cv=5)
    # scoring='roc_auc' ---> 'log_loss'
    gsearch1.fit(X_train, label)
    print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    '''
    # Create a Random Forest Model.
    print("Training the RF model ...")
    model = RandomForestClassifier(n_estimators=350,
                                   oob_score=True,
                                   min_samples_split=130,
                                   min_samples_leaf=20,
                                   random_state=10,
                                   max_depth=7,
                                   max_features='sqrt')
    model = model.fit(X_train, label)
    print model.oob_score_
    print model.feature_importances_

    # Predict the Competition Data with the newly trained model
    print("Predicting the Competition Data...")
    y_test = model.predict_proba(X_test)
    pred = y_test[:, 1]  # Get the probabilty of being 1.
    pred_df = pd.DataFrame(data={'Target': pred})
    submissions = pd.DataFrame(ID).join(pred_df)

    # Write the CSV File and Get Ready for Submission
    # Save the predictions out to a CSV file
    print("Writing predictions to abai_submissions.csv")
    submissions.to_csv("submissions_RF4.csv", index=False)
    
    ##
    #################################

# Here the main program.


if __name__ == '__main__':
    main()
