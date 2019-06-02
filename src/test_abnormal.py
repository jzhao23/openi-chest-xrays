import os
import numpy as np

from tensorflow.keras.models import load_model
from utils import load_X_and_Y, load_X_descr, most_common_list
from sklearn.metrics import roc_auc_score

model_path = "models/cnn_heuristic.h5"

class TestAbnormal():
    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset = np.load(os.path.join(curr_dir, "../data/dataset.npy"))

    def test_top_eight_abnormalities(self, model_path):
        # load model
        model = load_model(model_path)

        # load data
        X, Y = load_X_and_Y(test_only=True)
        _, _, x_test = X
        _, _, y_test = Y
        print("x_test shape", x_test.shape)
        print("y_test shape", y_test.shape)

        X = load_X_descr(test_only=True)
        _, _, x_test_all_descr = X

        # segment test set based on description and obtain accuracies
        # for each description
        roc_auc_scores = {}
           
        for descr in most_common_list(self.dataset, num_most_common=9):
            if descr == "normal":
                continue
            
            x_test_descr, y_test_descr = [], []
            for idx in range(len(x_test)):
                if x_test_all_descr[idx] == descr:
                    x_test_descr.append(x_test[idx])
                    y_test_descr.append(y_test[idx])
            x_test_descr = np.array(x_test_descr).reshape((-1, 224, 224, 3))
            y_test_descr = np.array(y_test_descr).reshape((-1, 1))

            preds = model.predict(x_test_descr).reshape((-1,1))
            """preds_list = []
            for pred in preds:
                preds_list.append(pred)
            y_list = []
            for label in y_test_descr:
                y_list.append(label)
            print(descr + " predictions: ", preds_list)
            print(descr + "labels: ", y_list)
            print("Performing Heuristic!")
            new_preds_list = []
            new_y_list = []
            for i in range(len(preds_list)):
                if preds_list[i] > 0.4 and preds_list[i] < 0.6:
                    continue
                new_preds_list.append(preds_list[i])
                new_y_list.append(y_list[i])
            #x_test_descr = np.array(x_test_descr).reshape((-1, 224, 224, 3))
            y_test_descr = np.array(new_y_list).reshape((-1, 1))
            preds = np.array(new_preds_list).reshape((-1, 1))
            print(descr + " formatted predictions: ", preds)
            print(descr + " formatted labels: ", y_test_descr)"""
            preds = preds.reshape((-1,)).tolist()
            y_test_descr = y_test_descr.reshape((-1,)).tolist()
            print("preds: ", preds)
            print("y_test_labels: ", y_test_descr)
            
            #roc_auc_scores[descr] = roc_auc_score(y_test_descr, preds)

            print(descr, roc_auc_scores[descr])

if __name__ == "__main__":
    test = TestAbnormal()
    test.test_top_eight_abnormalities(model_path)




