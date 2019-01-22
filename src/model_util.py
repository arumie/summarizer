import os
import tensorflow as tf
from tensorflow import keras
from my_flags import FLAGS
from data_util import Data


class NN:
    def __init__(self, feature_set=FLAGS['all_features']):
        """
        :type feature_set: FLAGS['all_features'], FLAGS['sent_embed']
        """
        self.feature_set = feature_set

        if not os.path.exists(FLAGS.model_checkpoints_dir): os.makedirs(FLAGS.model_checkpoints_dir)


        print('==========================================================')
        print('|                    Setting up data                     |')
        print('==========================================================')
        self.data = Data(feature_set=feature_set.default)

        print('==========================================================')
        print('|                     Setup model                        |')
        print('==========================================================')

        self.input_length = FLAGS[feature_set.name + '_input_length'].value
        self.model = self.create_model()

        print('==========================================================')
        print('|                       Training                         |')
        print('==========================================================')

        self.history = self.train()

        print('==========================================================')
        print('|                       Evaluate                         |')
        print('==========================================================')

        self.evaluate_model()

    def create_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.Dense(self.input_length/2, activation=tf.nn.sigmoid, input_shape=(self.input_length,)),
            keras.layers.Dense(self.input_length/4, activation=tf.nn.relu, input_shape=(self.input_length,)),
            keras.layers.Dense(1, activation=tf.nn.softmax)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=['accuracy'])

        model.summary()
        return model

    def train(self):
        with tf.device('/gpu:0'):
            self.model.fit(
                self.data.train_data,
                self.data.train_labels,
                epochs=100,
                validation_data=(self.data.val_data, self.data.val_labels),
                batch_size=10000,
                verbose=2)

    def evaluate_model(self):
        print(self.model.evaluate(self.data.test_data, self.data.test_labels))

    def create_callback(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            FLAGS.model_checkpoints_dir,
            save_best_only=True,
            save_weights_only=True,
            verbose=1)
        return cp_callback

    def load_model_and_evaluate(self):
        model = self.create_model()
        model.load_weights(FLAGS.model_checkpoints_dir)


