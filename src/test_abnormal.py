from tensorflow.keras.models import load_model
from utils import load_X_and_Y

model_path = "~/openi-chest-xrays/src/cnn_baseline.h5"

class TestAbnormal():
    def test_top_eight_abnormalities(self, model_path):
        # load model
        model = load_model(model_path)

        # load data
        X, Y = load_X_and_Y()
        _, _, x_test = X
        _, _, y_test = Y

        # test model on abnormal categories

        print(len(x_test))
        print(len(y_test))

        count = 0
        for x_test_data in x_test:
            if count > 5: 
                break
            print(x_test_data)
            count += 1

        count = 0
        for y_test_label in y_test:
            if count > 5: 
                break
            print(y_test_label)
            count += 1

if __name__ == "__main__":
    test = TestAbnormal()
    test.test_top_eight_abnormalities(model_path)




