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
        if not os.path.exists(FLAGS.training_data_dir): os.makedirs(FLAGS.training_data_dir)
        if not os.path.exists(FLAGS.validation_data_dir): os.makedirs(FLAGS.validation_data_dir)
        if not os.path.exists(FLAGS.test_data_dir): os.makedirs(FLAGS.test_data_dir)

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
            annotators="tokenize ssplit pos".split())
        self.rouge_evaluator = rouge.Rouge(
            metrics=['rouge-n', 'rouge-l'],
            max_n=2,
            apply_avg=True,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True)
        self.word2vec_model = None


        # ----------------------- EXTRACT FEATURES -----------------------
        if feature_set == FLAGS.doc_features or feature_set == FLAGS.all_features:
            for i in range(1, 8):
              self.extract_document_features(train_stories[10000*(i-1):10000*i], FLAGS.training_data_dir, name='document_features' + str(i))
            self.extract_document_features(train_stories[70000:], FLAGS.training_data_dir, name='document_features8')
            train_doc_features = self.load_training_data_document_features()
            val_doc_features = self.extract_document_features(val_stories, FLAGS.validation_data_dir)
            test_doc_features = self.extract_document_features(test_stories, FLAGS.test_data_dir)

        # ----------------------- CREATE LABELS -----------------------
        train_doc_labels = self.label_dataset(train_stories, FLAGS.training_data_dir)
        val_doc_labels = self.label_dataset(val_stories, FLAGS.validation_data_dir)
        test_doc_labels = self.label_dataset(test_stories, FLAGS.test_data_dir)

        # ----------------------- CREATE DOCUMENT EMBEDDINGS -----------------------

        if feature_set == FLAGS.sent_embed or feature_set == FLAGS.all_features:
            for i in range(1, 8):
              self.create_word_embeddings(train_stories[10000*(i-1):10000*i], FLAGS.training_data_dir, name='document_embeddings' + str(i))
            self.create_word_embeddings(train_stories[70000:], FLAGS.training_data_dir, name='document_embeddings8')
            train_doc_embeddings = self.load_training_data_document_embeddings()
            val_doc_embeddings = self.create_word_embeddings(val_stories, FLAGS.validation_data_dir)
            test_doc_embeddings = self.create_word_embeddings(test_stories, FLAGS.test_data_dir)


        # ----------------------- CREATE FEATURE SETS AND CORRESPONDING LABELS -----------------------
        if feature_set == FLAGS.sent_embed:
            self.setup_for_rouge_eval(test_doc_embeddings, [])
            (self.train_data, self.train_labels) = self.setup_input_data_embed_only(train_doc_embeddings, train_doc_labels, 'training')
            (self.val_data, self.val_labels) = self.setup_input_data_embed_only(val_doc_embeddings, val_doc_labels, 'validation')
            (self.test_data, self.test_labels) = self.setup_input_data_embed_only(test_doc_embeddings, test_doc_labels, 'testing')
        if feature_set == FLAGS.doc_features:
            self.setup_for_rouge_eval([], test_doc_features)
            (self.train_data, self.train_labels) = self.setup_input_data_only_doc_features(train_doc_features, train_doc_labels, 'training')
            (self.val_data, self.val_labels) = self.setup_input_data_only_doc_features(val_doc_features, val_doc_labels, 'validation')
            (self.test_data, self.test_labels) = self.setup_input_data_only_doc_features(test_doc_features, test_doc_labels, 'testing')
        if feature_set == FLAGS.all_features:
            self.setup_for_rouge_eval(test_doc_embeddings, test_doc_features)
            (self.train_data, self.train_labels) = self.setup_input_data_all_features(train_doc_embeddings, train_doc_features, train_doc_labels, 'training')
            (self.val_data, self.val_labels) = self.setup_input_data_all_features(val_doc_embeddings, val_doc_features, val_doc_labels, 'validation')
            (self.test_data, self.test_labels) = self.setup_input_data_all_features(test_doc_embeddings, test_doc_features, test_doc_labels, 'testing')

        self.test_stories = test_stories

    def setup_for_rouge_eval(self, test_doc_embeddings, test_doc_features):
        print('================ Setting up for Rouge ================')
        # Baseline is first three sentences
        self.baseline_predictions = []

        self.evaluation_doc_features = []
        if self.feature_set == FLAGS.sent_embed or self.feature_set == FLAGS.all_features:
            print('------------ Document embeddings --------------')
            self.evaluation_doc_features = test_doc_embeddings

            print('------------ Making base predictions --------------')
            i = 0
            for doc in test_doc_embeddings:
                self.baseline_predictions.append([1 for i in range(0, len(doc))])
                i += 1
                if i % 1000 == 0:
                    print(i)
        doc_features_keys = ['DOC_SENT', 'DOC_WRD', 'DOC_CHAR', 'DOC_AVG_SENT_LEN']
        if self.feature_set == FLAGS.doc_features:
            i = 0
            print('--------------- Stitching doc features together -----------------')
            for doc_feature in test_doc_features:
                doc_values = [doc_feature[key] for key in doc_features_keys]
                self.evaluation_doc_features.append(
                    [np.concatenate((doc_values, [sent_position, sent_pos_tag, sent_len]))
                     for sent_position, sent_pos_tag, sent_len
                     in zip(doc_feature['SENT_POS'], doc_feature['SENT_POST_TAGS'], doc_feature['SENT_LEN'])])

                i += 1
                if i % 1000 == 0:
                    print(i)
            print('------------ Making base predictions --------------')
            for doc in test_doc_features:
                self.baseline_predictions.append([1 if i < 3 else 0 for i in range(0, len(doc['SENT_LEN']))])
        if self.feature_set == FLAGS.all_features:
            self.evaluation_doc_features = []
            for doc_embedding, doc_feature in zip(test_doc_embeddings, test_doc_features):
                doc_all_features = []
                doc_values = [doc_feature[key] for key in doc_features_keys]
                sent_positions, sent_pos_tags, sent_lens = doc_feature['SENT_POS'], doc_feature['SENT_POST_TAGS'], \
                                                           doc_feature['SENT_LEN']
                for sent_embedding, sent_pos, sent_pos_tag, sent_len in zip(doc_embedding, sent_positions,
                                                                            sent_pos_tags, sent_lens):
                    if np.shape(sent_embedding) == (300,):

                        feature_values = np.concatenate((doc_values, [sent_pos, sent_pos_tag, sent_len]), axis=0)
                        doc_all_features.append(np.concatenate((sent_embedding, feature_values), axis=0))

                self.evaluation_doc_features.append(doc_all_features)
            print(self.evaluation_doc_features[0][0])

        print(self.evaluation_doc_features[0])


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
            np.save(dest +  name, document_embeddings)
        else:
            print('------------------------ Loading from previous file: {0}{1}.npy------------------------'.format(dest, name))
            document_embeddings = np.load(dest +  name + '.npy')
        print(np.shape(document_embeddings))
        print(np.shape(document_embeddings[0]))
        return document_embeddings

    def extract_document_features(self, data_stories, dest, name='document_features'):
        print('================ Extracting features from documents for {0}{1}.npy ================'.format(dest, name))
        # Document length, Chars, words and sentences
        # Average sentence length
        # Sentence length - Chars and Words
        # Sentence position
        # NER - nr of entities, DATE, TIME, LOCATION, ORGANIZATION, NUMBER, CITY, COUNTRY, MENTIONS
        # Sentence Nr of POS tags - NNP, NNPS
        document_features = []
        if self.new_stories or not os.path.exists(dest + name + '.npy'):
            print('------------------------ Creating list of document features ------------------------')
            i = 0
            for s in data_stories:
                with open(FLAGS.cnn_stories_dir + s, 'rb') as story:
                    r = story.read().decode('utf-8')
                    body = r.split('@highlight')[0]
                    ann = self.nlp_client.annotate(body)
                    if len(ann.sentence) == 0:
                        print('No Sentences: {0}'.format(s))
                    else:
                        i += 1
                        doc_nr_sent = len(ann.sentence)
                        doc_nr_words = 0
                        doc_nr_chars = len(body)
                        doc_avg_sent_len = np.sum([len(sent.token) for sent in ann.sentence])/doc_nr_sent
                        # sent_ner = []
                        sent_pos = [i for i in range(1, len(ann.sentence) + 1)]
                        sent_pos_tags = []
                        sent_len = []

                        for sentence in ann.sentence:
                            # ner = {'DATE': 0, 'TIME': 0, 'NUMBER': 0, 'LOCATION': 0, 'PERSON': 0, 'CITY': 0, 'COUNTRY': 0}
                            pos_tags = ['NNP', 'NNPS']
                            nr_pos_tags = 0
                            sent_len.append(len(sentence.token))
                            for token in sentence.token:
                                doc_nr_words += 1
                                if token.pos in pos_tags:
                                    nr_pos_tags += 1
                            # for mention in sentence.mentions:
                            #     if mention.ner in ner.keys():
                            #         ner[mention.ner] += 1
                            # sent_ner.append([ner[n] for n in ner.keys()])
                            sent_pos_tags.append(nr_pos_tags)
                        document_features.append(
                            {'DOC_SENT': doc_nr_sent,
                             'DOC_WRD': doc_nr_words,
                             'DOC_CHAR': doc_nr_chars,
                             'DOC_AVG_SENT_LEN':  doc_avg_sent_len,
                             # 'SENT_NER': sent_ner,
                             'SENT_POS': sent_pos,
                             'SENT_POS_TAGS': sent_pos_tags,
                             'SENT_LEN': sent_len})
                        if i % 1000 == 0:
                            print('{0} documents done!'.format(i))
            print(document_features[0])
            print(np.shape(document_features))
            np.save(dest + name, document_features)
        else:
            print('------------------------ Loading from previously made file: {0}document_feature.npy ------------------------'.format(dest))
            document_features = np.load(dest + name + '.npy')
            print(np.shape(document_features))
            print(np.shape(document_features[0]))
        return document_features


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

    def load_training_data_document_features(self):
        print('================ Stitching document features for training data together ================')
        train_doc_features = []
        for i in range(1,9):
            next_doc_embeds = np.load(FLAGS.training_data_dir + 'document_features' + str(i) + '.npy')
            for doc in next_doc_embeds:
                train_doc_features.append(doc)
            print(np.shape(train_doc_features))

        print(np.shape(train_doc_features))
        print(np.shape(train_doc_features[0]))
        return train_doc_features

    def setup_input_data_embed_only(self, doc_embeddings, doc_labels, type):
        print('================ Readying data for {0} ================'.format(type))
        data, labels = [], []
        for doc_feature, doc_label in zip(doc_embeddings, doc_labels):
            assert len(doc_feature) == len(doc_label)
            for sent_feature, sent_label in zip (doc_feature, doc_label):
                if np.shape(sent_feature) == (300,):
                    data.append(sent_feature)
                    labels.append(sent_label)
        print(np.shape(data))
        print(np.shape(data[0]))
        print(np.shape(labels))
        return data, labels

    def setup_input_data_all_features(self, doc_embeddings, doc_features, doc_labels, type):
        print('================ Readying data for {0} ================'.format(type))
        doc_features_keys = ['DOC_SENT', 'DOC_WRD', 'DOC_CHAR', 'DOC_AVG_SENT_LEN']
        print(doc_features[0])
        data, labels = [], []
        for doc_embedding, doc_feature, doc_label in zip(doc_embeddings, doc_features, doc_labels):
            assert len(doc_embedding) == len(doc_label)
            doc_values = [doc_feature[key] for key in doc_features_keys]
            sent_positions, sent_pos_tags, sent_lens = doc_feature['SENT_POS'], doc_feature['SENT_POST_TAGS'], doc_feature['SENT_LEN']
            for sent_embedding, sent_pos, sent_pos_tag, sent_len, sent_label in zip(doc_embedding, sent_positions, sent_pos_tags, sent_lens, doc_label):
                if np.shape(sent_embedding) == (300,):
                    feature_values = np.concatenate((doc_values, [sent_pos, sent_pos_tag, sent_len]), axis=0)
                    data.append(np.concatenate((np.multiply(sent_embedding), feature_values), axis=0))
                    labels.append(sent_label)
        print(data[0])
        print(np.shape(data[0]))
        print(np.shape(labels))
        return data, labels

    def setup_input_data_only_doc_features(self, doc_features, doc_labels, type):
        print('================ Readying data for {0} ================'.format(type))
        doc_features_keys = ['DOC_SENT', 'DOC_WRD', 'DOC_CHAR', 'DOC_AVG_SENT_LEN']
        print(doc_features[0])
        data, labels = [], []
        for doc_feature, doc_label in zip(doc_features, doc_labels):
            doc_values = [doc_feature[key] for key in doc_features_keys]
            sent_positions, sent_pos_tags, sent_lens = doc_feature['SENT_POS'], doc_feature['SENT_POST_TAGS'], doc_feature['SENT_LEN']
            for sent_pos, sent_pos_tag, sent_len, sent_label in zip(sent_positions, sent_pos_tags, sent_lens, doc_label):
                    feature_values = np.concatenate((doc_values, [sent_pos, sent_pos_tag, sent_len]), axis=0)
                    data.append(feature_values)
                    labels.append(sent_label)
        print(data[0])
        print(np.shape(data[0]))
        print(np.shape(labels))
        return data, labels



