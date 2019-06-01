import keras
import numpy as np

from datetime import datetime
from tensorflow.keras import optimizers, losses
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import roc_auc_score

from utils import img_dims

batch_size = 64
num_epochs = 15
lr = 0.0001
lr_decay = 0.005

class CNN(object):
    def __init__(self,
                 img_dims=img_dims,
                 lr=lr, lr_decay=lr_decay,
                 save_name='CNN_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')):
        self._save_name = save_name
        
        model = ResNet50(include_top=False, weights='imagenet',
                         pooling='max', input_shape=img_dims)
        pred = Dense(1, activation='sigmoid')(model.layers[-1].output)
        self._model = Model(inputs=model.input, outputs=pred)
        self._model.compile(loss=losses.binary_crossentropy,
            optimizer=optimizers.Adam(lr=lr, decay=lr_decay),
            metrics=['accuracy'])

    def save(self):
        self._model.save(self._save_name + '.h5')

    def fit(self,
            x_train, y_train,
            x_dev, y_dev,
            batch_size=batch_size,
            num_epochs=num_epochs,
            save=True):
        self._model.fit(x_train, y_train,
                        validation_data=(x_dev, y_dev),
                        batch_size=batch_size,
                        epochs=num_epochs)
        if save:
            self.save()

    def predict(self, x):
        return self._model.predict(x)

    def evaluate(self, x_test, y_test, verbose):
        preds = self.predict(x_test).reshape((-1, 1))
        if verbose == 1:
            print("Performing Heuristic!")
            new_preds = []
            new_y_test = []
            for idx,pred in enumerate(preds):
                if pred > 0.4 and pred < 0.6:
                    continue 
                new_preds.append(pred)
                new_y_test.append(y_test[idx])
            roc_auc = roc_auc_score(new_y_test, new_preds)
        else:
            roc_auc = roc_auc_score(y_test, preds)
        return roc_auc