import os
import corenlp
import numpy as np
import rouge
import gensim
from my_flags import FLAGS
from sklearn.model_selection import train_test_split

PAD_ID = 0
UNK_ID = 1

class Data:
    def __init__(self, reset=False, feature_set=FLAGS.all_features):

        #----------------------- FLAGS -----------------------

        self.new_stories = reset
        self.feature_set = feature_set

        #----------------------- DATA VARIABLES -----------------------

        self.train_data = []
        self.val_data = []
        self.test_data = []

        #----------------------- LABEL VARIABLES -----------------------

        self.train_labels = []
        self.val_labels = []
        self.test_labels = []

        stories, train_stories, val_stories, test_stories = self.split_dataset(reset)

        #----------------------- TOOLS -----------------------

        self.nlp_client = corenlp.CoreNLPClient(
            endpoint="http://localhost:8000",
            annotators="tokenize ssplit".split())
        self.rouge_evaluator = rouge.Rouge(
            metrics=['rouge-n', 'rouge-l', 'rouge-w'],
            max_n=2,
            apply_avg=True,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True)
        self.word2vec_model = None

        # ----------------------- CREATE LABELS -----------------------
        train_doc_labels = self.label_dataset(train_stories, FLAGS.training_data_dir)
        val_doc_labels = self.label_dataset(val_stories, FLAGS.validation_data_dir)
        test_doc_labels = self.label_dataset(test_stories, FLAGS.test_data_dir)

        # ----------------------- CREATE DOCUMENT EMBEDDINGS -----------------------
        for i in range(1,8):
            self.create_word_embeddings(train_stories[10000*(i-1):10000*i], FLAGS.training_data_dir, name='document_embeddings' + str(i))
        self.create_word_embeddings(train_stories[70000:], FLAGS.training_data_dir, name='document_embeddings8')
        train_doc_embeddings = self.load_training_data_document_embeddings()
        val_doc_embeddings = self.create_word_embeddings(val_stories, FLAGS.validation_data_dir)
        test_doc_embeddings = self.create_word_embeddings(test_stories, FLAGS.test_data_dir)

        # ----------------------- CREATE FEATURE SETS AND CORRESPONDING LABELS -----------------------
        if feature_set == FLAGS.sent_embed:
            (self.train_data, self.train_labels) = self.setup_input_data_embed_only(train_doc_embeddings, train_doc_labels, 'training')
            (self.val_data, self.val_labels) = self.setup_input_data_embed_only(val_doc_embeddings, val_doc_labels, 'validation')
            (self.test_data, self.test_labels) = self.setup_input_data_embed_only(test_doc_embeddings, test_doc_labels, 'testing')

    def split_dataset(self, reset):
        print('================ Splitting data into training, validation and test set ================')

        if reset or not os.path.exists(FLAGS.training_data_dir + 'stories.npy') or not os.path.exists(FLAGS.validation_data_dir + 'stories.npy') or not os.path.exists(FLAGS.test_data_dir + 'stories.npy'):
            print('------------------------ Re-splitting the data ------------------------')

            stories = np.random.permutation(np.array(os.listdir(FLAGS.cnn_stories_dir)))
            train_stories, rest = train_test_split(stories, train_size=FLAGS.training_data_percent / 100)
            val_stories, test_stories = train_test_split(rest, test_size=.5)
            test_stories = test_stories

            np.save(FLAGS.training_data_dir + 'stories.npy', train_stories)
            np.save(FLAGS.validation_data_dir + 'stories.npy', val_stories)
            np.save(FLAGS.test_data_dir + 'stories.npy', test_stories)

            self.new_stories = True

        else:
            print('------------------------ Loading from previous files ------------------------')
            train_stories = np.load(FLAGS.training_data_dir + 'stories.npy')
            val_stories = np.load(FLAGS.validation_data_dir + 'stories.npy')
            test_stories = np.load(FLAGS.test_data_dir + 'stories.npy')
            stories = np.append(np.append(train_stories, val_stories), test_stories)

            self.new_stories = False
        print(np.shape(train_stories))
        print(np.shape(val_stories))
        print(np.shape(test_stories))
        return stories, train_stories, val_stories, test_stories

    def label_dataset(self, data_stories, dest):
        print('================ Labeling "{0}" data ================'.format(dest))
        if self.new_stories or not os.path.exists(dest + 'labels.npy'):
            print('------------------------ Re-labeling the data ------------------------')
            labels = []
            stories = []
            i = 0
            for s in data_stories:
                with open(FLAGS.cnn_stories_dir + s, 'rb') as story:
                    r = story.read().decode('utf-8')
                    highlights = [h.replace('\n\n', '') for h in r.split('@highlight')[1:]]
                    body = r.split('@highlight')[0]
                    ann = self.nlp_client.annotate(body)
                    sentences = [self.ann2sentence(s) for s in ann.sentence]
                    if len(sentences) == 0:
                        continue
                    else:
                        labels.append(self.label_document(sentences, highlights))
                        stories.append(s)
                        i += 1
                        if i % 10 == 0:
                            print('{0} documents done!'.format(np.where(data_stories == s)))
            np.save(dest + 'labels', labels)
            if len(stories) < len(data_stories):
                print('{0} stories were empty and were removed'.format(len(data_stories) - len(stories)))
                np.save(dest + 'stories', stories)
        else:
            print('------------------------ Loading from previous files ------------------------')
            labels = np.load(dest + 'labels.npy')
        print(np.shape(labels))
        print(np.shape(labels[0]))
        return labels


    def get_highest_rouge(self, highlight, sentences):
        highest_id = 0
        current_highest_avg = -1000
        for i in range(len(sentences)):
            scores = self.rouge_evaluator.get_scores([sentences[i]], [highlight])
            f_scores = [scores[key]['f'] for key in scores.keys()]
            avg_rouge = np.average(f_scores)
            if avg_rouge > current_highest_avg:
                current_highest_avg = avg_rouge
                highest_id = i
        return highest_id

    def label_document(self, sentences, highlights):
        labels = np.zeros(len(sentences), dtype=np.int)
        for h in highlights:
            highest_rouge = self.get_highest_rouge(h, sentences)
            labels[highest_rouge] = int(1)
        return labels 

    def ann2sentence(self, ann_sentence):
        result = ''
        for t in ann_sentence.token:
            result += t.word + ' '
        return result

    def create_word_embeddings(self, data_stories, dest, name='document_embeddings'):
        print('================ Creating word embeddings for {0} ================'.format(dest))
        if self.new_stories or not os.path.exists(dest +  name + '.npy'):
            if self.word2vec_model is None:
                print('------------------------ Loading word2vec model ------------------------')
                self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.google_news_word2vec, binary=True)
            print('------------------------ Creating list of sentence embeddings ------------------------')
            document_embeddings = []
            i = 0
            for s in data_stories:
                with open(FLAGS.cnn_stories_dir + s, 'rb') as story:
                    r = story.read().decode('utf-8')
                    body = r.split('@highlight')[0]
                    ann = self.nlp_client.annotate(body)
                    if len(ann.sentence) == 0:
                        print('No Sentences: {0}'.format(s))
                    else:
                        document_embeddings.append(self.embed_document(ann.sentence))
                        i += 1
                        if i % 5000 == 0:
                            print('{0} documents done!'.format(i))
            print(document_embeddings[:1])
            np.save(dest +  name, document_embeddings)
        else:
            print('------------------------ Loading from previous file: {0}{1}.npy------------------------'.format(dest, name))
            document_embeddings = np.load(dest +  name + '.npy')
        print(np.shape(document_embeddings))
        print(np.shape(document_embeddings[0]))
        return document_embeddings

    def embed_document(self, sentences):
        sentence_embeddings = []
        for sentence in sentences:
            sentence_embeddings.append(self.embed_sentence(sentence))
        return sentence_embeddings

    def embed_sentence(self, sentence):
        word_vectors = []
        for token in sentence.token:
            if str(token.value) in self.word2vec_model:
                word_vectors.append(self.word2vec_model[str(token.value)])
        return np.mean(word_vectors, axis=0)

    def load_training_data_document_embeddings(self):
        print('================ Stitching word embeddings for training data together ================')
        train_doc_embeds = []
        for i in range(1,9):
            next_doc_embeds = np.load(FLAGS.training_data_dir + 'document_embeddings' + str(i) + '.npy')
            for doc in next_doc_embeds:
                train_doc_embeds.append(doc)
            print(np.shape(train_doc_embeds))

        print(np.shape(train_doc_embeds))
        print(np.shape(train_doc_embeds[0]))
        return train_doc_embeds

    def setup_input_data_embed_only(self, doc_features, doc_labels, type):
        print('================ Readying data for {0} ================'.format(type))
        data, labels = [], []
        for doc_feature, doc_label in zip(doc_features, doc_labels):
            assert len(doc_feature) == len(doc_label)
            for sent_feature, sent_label in zip (doc_feature, doc_label):
                data.append(sent_feature)
                labels.append(sent_label)
        print(np.shape(data))
        print(np.shape(data[0]))
        print(data[0])
        print(np.shape(labels))
        print(labels[0])
        return data, labels

def main():
    data = Data(feature_set=FLAGS.sent_embed)

if __name__ == '__main__':
    main()