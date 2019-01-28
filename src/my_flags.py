import numpy as np
import tensorflow as tf

## DIRECTORIES

tf.app.flags.DEFINE_string("data_processed_dir", '../data/processed/', "Directory for the processed data")
tf.app.flags.DEFINE_string("training_data_dir", '../data/processed/train/', "Directory for the processed training data")
tf.app.flags.DEFINE_string("test_data_dir", '../data/processed/test/', "Directory for the processed test data")
tf.app.flags.DEFINE_string("validation_data_dir", '../data/processed/val/', "Directory for the processed val data")
tf.app.flags.DEFINE_string("cnn_stories_dir", '../../cnn/stories/', "Directory for the CNN stories")
tf.app.flags.DEFINE_string("dm_stories_dir", '../../dailymail/stories/', "Directory for the DailyMail stories")
tf.app.flags.DEFINE_string("google_news_word2vec", '../../GoogleNews-vectors-negative300.bin', "Location for the Google News trained word2vec")

tf.app.flags.DEFINE_string("model_checkpoints_dir", '../data/model/stories/', "Directory for the CNN stories")

## PARAMETERS

tf.app.flags.DEFINE_integer('training_data_percent', 80, 'The percentage of the data that should be training data')
tf.app.flags.DEFINE_integer('test_data_percent', 10, 'The percentage of the data that should be training data')
tf.app.flags.DEFINE_integer('val_data_percent', 10, 'The percentage of the data that should be training data')


tf.app.flags.DEFINE_integer('sent_embed', 0, 'Specifies that the input data only consists of sentence embeddings')
tf.app.flags.DEFINE_integer('sent_embed_input_length', 300, 'Input length of the features set')

tf.app.flags.DEFINE_integer('doc_features', 1, 'The input data consisting of the features from the documents')
tf.app.flags.DEFINE_integer('doc_features_input_length', 7, 'Input length of the features set')

tf.app.flags.DEFINE_integer('all_features', 2, 'The input data consisting of all the features')
tf.app.flags.DEFINE_integer('all_features_input_length', 307, 'Input length of the features set')

## OTHER

tf.app.flags.DEFINE_string("processing_train_data", 'train', "Processing Training data")
tf.app.flags.DEFINE_string("processing_test_data", 'test', "Processing Test data")
tf.app.flags.DEFINE_string("processing_val_data", 'val', "Processing Validation data")


FLAGS = tf.app.flags.FLAGS