#!/usr/bin/env python


from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
import matplotlib.pyplot as plt


def main():
    print "loading data:"
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    Y = np.load('Label.npy')
    # n = len(X_train[0])
    '''
    print "Decompositing:"
    pca = PCA(n_components=36)
    new_train = pca.fit_transform(X_train)
    new_test = pca.transform(X_test)

    scaler = MinMaxScaler()
    scaler.fit(new_train)
    new_train = scaler.transform(new_train)
    new_test = scaler.transform(new_test)
    '''
    print 'fitting..........'
    # Gaussian-Based model
    new_train = X_train
    new_test = X_test
    # clf1 = EllipticEnvelope()
    clf1 = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
    # clf1.fit(new_train[:, :4])
    clf1.fit(new_train)

    # clf2 = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
    # clf2.fit(new_train[:, :4])

    print 'predicting the train..........'
    # y_pred_train1 = clf1.predict(new_train[:, :4])
    y_pred_train1 = clf1.predict(new_train)
    # y_pred_train2 = clf2.predict(new_train[:, :4])
    # y_pred_train = y_pred_train1 * y_pred_train2
    y_pred_train = y_pred_train1

    final_train = np.array([])

    for i in range(y_pred_train.shape[0]):
        if y_pred_train[i] == 1:
            if len(final_train) == 0:
                final_train = new_train[i]
                final_label = Y[i]
            else:
                final_train = np.vstack((final_train, new_train[i]))
                final_label = np.vstack((final_label, Y[i]))

    print X_train.shape, final_train.shape

    normal = X_train[y_pred_train == 1]
    abnormal = X_train[y_pred_train == -1]
    plt.plot(normal[:, 1], normal[:, 3], 'bx')
    plt.plot(abnormal[:, 1], abnormal[:, 3], 'ro')
    plt.show()
    np.save('New_train4', final_train)
    np.save('Label4', final_label)
    np.save('New_test4', new_test)
    # pca = PCA(n_components=9)
    # pca.fit(X_train)
    # print pca.explained_variance_ratio_


if __name__ == '__main__':
    main()
