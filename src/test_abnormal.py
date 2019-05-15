from tensorflow.keras.models import load_model
from utils import load_X_and_Y

def test_abnormal(model_path):
    model = load_model(model_path)
