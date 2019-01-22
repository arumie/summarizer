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
## PARAMETERS

tf.app.flags.DEFINE_integer('training_data_percent', 80, 'The percentage of the data that should be training data')
tf.app.flags.DEFINE_integer('test_data_percent', 10, 'The percentage of the data that should be training data')
tf.app.flags.DEFINE_integer('val_data_percent', 10, 'The percentage of the data that should be training data')


## OTHER

tf.app.flags.DEFINE_string("processing_train_data", 'train', "Processing Training data")
tf.app.flags.DEFINE_string("processing_test_data", 'test', "Processing Test data")
tf.app.flags.DEFINE_string("processing_val_data", 'val', "Processing Validation data")


FLAGS = tf.app.flags.FLAGS