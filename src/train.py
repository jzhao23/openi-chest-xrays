import sys
import numpy as np

from cnn import CNN
from utils import load_X_and_Y, load_X_descr

def train(holdout=False, holdout_list=[]):
    # load data
    X, Y = load_X_and_Y()
    x_train, x_dev, x_test = X
    y_train, y_dev, y_test = Y

    # holdout training and dev data if requested
    if holdout:
        X_descr = load_X_descr()
        x_train_all_descr, x_dev_all_descr, _ = X_descr

        holdout_x_train = []
        holdout_y_train = []
        for idx in range(len(x_train)):
            if x_train_all_descr[idx] not in holdout_list:
                holdout_x_train.append(x_train[idx])
                holdout_y_train.append(y_train[idx])

        holdout_x_dev = []
        holdout_y_dev = []
        for idx in range(len(x_dev)):
            if x_dev_all_descr[idx] not in holdout_list:
                holdout_x_dev.append(x_dev[idx])
                holdout_y_dev.append(y_dev[idx])
        
        x_train = np.array(holdout_x_train).reshape((-1, 224, 224, 3))
        y_train = np.array(holdout_y_train).reshape((-1, 1))

        x_dev = np.array(holdout_x_dev).reshape((-1, 224, 224, 3))
        y_dev = np.array(holdout_y_dev).reshape((-1, 1))
    
    # train model
    model = CNN()
    model.fit(x_train, y_train, x_dev, y_dev, save=True)
    model.evaluate(x_test, y_test, is_test=False)

if __name__ == "__main__":
    holdout_list = ["Calcinosis", "Opacity", "Thoracic Vertebrae", "Calcified Granuloma"]
    train(holdout=True, holdout_list=holdout_list)
