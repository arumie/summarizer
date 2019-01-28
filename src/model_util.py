import os
import tensorflow as tf
from tensorflow import keras
from my_flags import FLAGS
from data_util import Data
import numpy as np
from sklearn import svm
import pickle

class Model:
    def __init__(self, feature_set=FLAGS['all_features'], type='NN'):
        """
        :type feature_set: FLAGS['all_features'], FLAGS['sent_embed']
        """
        self.feature_set = feature_set
        self.type = type

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
        model = None
        if self.type == 'NN':
            model = tf.keras.models.Sequential([
                keras.layers.Dense(round(2*self.input_length/3, 0), activation=tf.nn.sigmoid, input_shape=(self.input_length,)),
                keras.layers.Dense(1, activation=tf.nn.softmax)
            ])

            model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001),
                          loss=tf.keras.losses.binary_crossentropy,
                          metrics=['accuracy'])

            model.summary()
        if self.type == 'SVM':
            model = svm.SVC(kernel='poly', C=0.001, verbose=True, max_iter=800)
        return model

    def train(self):
        if self.type == 'NN':
            with tf.device('/gpu:0'):
                self.model.fit(
                    np.array(self.data.train_data),
                    np.array(self.data.train_labels),
                    validation_data=(np.array(self.data.val_data), np.array(self.data.val_labels)),
                    batch_size=10000,
                    epochs=10,
                    callbacks=[self.create_callback()],
                    verbose=1)
        if self.type == 'SVM':
            self.model.fit(self.data.train_data, self.data.train_labels)
            pickle.dump(self.model, open('../data/model/svm_'+ self.feature_set.name + '.sav', 'wb'))

    def evaluate_model(self):
        print('============================ Calculate Baseline Rouge ==============================')

        self.calculate_avg_rouge(self.data.baseline_predictions, 'baseline')

        predictions = []

        if self.type == 'NN':
            print('============================ Calculate Model Accuracy and Rouge ==============================')
            print(self.model.evaluate(np.array(self.data.test_data), np.array(self.data.test_labels)))
            print(np.array(self.data.evaluation_doc_features[0]))
            for doc in self.data.evaluation_doc_features:
                predictions.append([self.model.predict(np.array(d.reshape(1,-1))) for d in doc])


        if self.type == 'SVM':
            print(self.model.score(self.data.test_data, self.data.test_labels))
            for doc_features in self.data.evaluation_doc_features:
                predictions.append([self.model.predict(np.array(sent_features)) for sent_features in doc_features])
            print(predictions[0])

        self.calculate_avg_rouge(predictions, self.type + '_' + self.feature_set.name)

    def calculate_avg_rouge(self, predictions, name):
        if name == 'baseline' and os.path.exists('../data/model/rouge_baseline.npy'):
            avg_rouge_scores = np.load('../data/model/rouge_baseline.npy')
        else:
            print(predictions[0])
            print(self.data.test_stories[0])
            rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
            i = 0
            for s, p in zip(self.data.test_stories, predictions):
                with open(FLAGS.cnn_stories_dir + s, 'rb') as story:
                    r = story.read().decode('utf-8')
                    highlights = [h.replace('\n\n', '') for h in r.split('@highlight')[1:]]
                    body = r.split('@highlight')[0]
                    ann = self.data.nlp_client.annotate(body)
                    sentences = [self.data.ann2sentence(s) for s in ann.sentence]
                    if len(sentences) == 0:
                        continue
                    else:
                        predicted_summaries = [sent for sent, pred in zip(sentences, p) if pred == 1]
                        scores = self.data.rouge_evaluator.get_scores(self.make_summary(predicted_summaries), self.make_summary(highlights))
                        for key in scores.keys():
                            if key in rouge_scores.keys():
                                rouge_scores[key].append(scores[key]['f'])
                        i += 1
                        if i % 1000 == 0:
                            print('{0} documents done!'.format(np.where(self.data.test_stories == s)))

            avg_rouge_scores = [[key, np.average(rouge_scores[key])] for key in rouge_scores.keys()]

            np.save('../data/model/rouge_' + name, avg_rouge_scores)
        print(avg_rouge_scores)



    def make_summary(self, string_list):
        result = ''
        for sent in string_list:
            result += ''.join(sent) + ' '
        return result
    def create_callback(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            FLAGS.model_checkpoints_dir,
            save_weights_only=True,
            verbose=1)
        return cp_callback

    def load_model_and_evaluate(self):
        model = self.create_model()
        model.load_weights(FLAGS.model_checkpoints_dir)


