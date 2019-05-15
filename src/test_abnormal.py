from tensorflow.keras.models import load_model
from utils import load_X_and_Y, load_X_descr

model_path = "models/cnn_baseline.h5"

class TestAbnormal():
    def test_top_eight_abnormalities(self, model_path):
        # load model
        model = load_model(model_path)

        # load data
        X, Y = load_X_and_Y(test_only=True)
        _, _, x_test = X
        _, _, y_test = Y

        X = load_X_descr(test_only=True)
        _, _, x_test_descr = X

        # test model on abnormal categories

        print(len(x_test))
        print(len(y_test))
        print(len(x_test_descr))

        count = 0
        for x in x_test_descr:
            if count > 5: 
                break
            print(x)
            count += 1

if __name__ == "__main__":
    test = TestAbnormal()
    test.test_top_eight_abnormalities(model_path)




