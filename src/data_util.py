import os
import corenlp
import numpy as np
import rouge
from my_flags import FLAGS
from sklearn.model_selection import train_test_split
import gensim


PAD_ID = 0
UNK_ID = 1


class Data:
    def __init__(self, reset=False):
        self.train_data = []
        self.val_data = []
        self.test_data = []

        self.train_labels = []
        self.val_labels = []
        self.test_labels = []

        self.new_stories = reset

        stories, train_stories, val_stories, test_stories = self.split_dataset(reset)

        self.nlp_client = corenlp.CoreNLPClient(endpoint="http://localhost:8000",
                                                annotators="tokenize ssplit".split())
        self.rouge_evaluator = rouge.Rouge(
            metrics=['rouge-n', 'rouge-l', 'rouge-w'],
            max_n=2,
            apply_avg=True,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True)

        self.train_labels = self.label_dataset(train_stories, FLAGS.training_data_dir)
        self.val_labels = self.label_dataset(val_stories, FLAGS.validation_data_dir)
        self.test_labels = self.label_dataset(test_stories, FLAGS.test_data_dir)

        self.word2vec_model = None

        self.create_word_embeddings(test_stories, FLAGS.test_data_dir)
        self.create_word_embeddings(val_stories, FLAGS.validation_data_dir)
        # self.create_word_embeddings(train_stories, FLAGS.training_data_dir)

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

    def create_word_embeddings(self, data_stories, dest):
        print('================ Creating word embeddings for {0} ================'.format(dest))
        print(os.path.exists(dest + 'document_embeddings.npy'))
        if self.new_stories or not os.path.exists(dest + 'document_embeddings.npy'):
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
            np.save(dest + 'document_embeddings', document_embeddings)
        else:
            print('------------------------ Loading from previous files ------------------------')
            document_embeddings = np.load(dest + 'document_embeddings.npy')
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


def main():
    data = Data()
    # nlp_client = corenlp.CoreNLPClient(endpoint="http://localhost:8000", annotators="tokenize ssplit".split())
    # rouge_evaluator = rouge.Rouge(
    #     metrics=['rouge-n', 'rouge-l', 'rouge-w'],
    #     max_n=2,
    #     apply_avg=True,
    #     alpha=0.5,  # Default F1_score
    #     weight_factor=1.2,
    #     stemming=True)
    #
    # print('==================================== Preparing data ====================================')
    #
    # print('Nr. of stories: {0}'.format(len(stories)))
    #
    #
    #
    # print('________________ Preparing the training data __________________')
    # print('Nr. of training data stories: {0}'.format(len(train_stories)))
    # all_text_file = open(FLAGS.training_data_dir + 'all_text', 'w')
    # labels_file = open(FLAGS.training_data_dir + 'labels.bin', 'w')
    # training_stories_file = open(FLAGS.training_data_dir + 'stories.bin', 'w')
    # i = 0
    # for s in train_stories:
    #     with open(FLAGS.cnn_stories_dir + s, 'rb') as story:
    #         r = story.read().decode('utf-8')
    #         highlights = [h.replace('\n\n','') for h in r.split('@highlight')[1:]]
    #         body = r.split('@highlight')[0]
    #         ann = nlp_client.annotate(body)
    #         sentences = [ann2sentence(s) for s in ann.sentence]
    #         if len(sentences) == 0:
    #             print('No Sentences: ')
    #         else:
    #             for l in label_document(sentences, highlights, rouge_evaluator):
    #                 labels_file.write(str(l) + ' ')
    #             labels_file.write('\n')
    #             for sentence in ann.sentence:
    #                 for token in sentence.token:
    #                     all_text_file.write(token.word + ' ')
    #             training_stories_file.write(s + '\n')
    #             all_text_file.write('\n')
    #             i += 1
    #             if i % 5000 == 0:
    #                 print('{0} documents done!'.format(i))
    #
    # print('________________ Word2Vec: Phrases __________________')
    # word2vec.word2phrase(FLAGS.training_data_dir + 'all_text',
    #                      FLAGS.training_data_dir + 'all_text_phrases', verbose=True)
    # print('________________ Word2Vec: Training __________________')
    # word2vec.word2vec(FLAGS.training_data_dir + 'all_text_phrases',
    #                   FLAGS.training_data_dir + 'word2vec_model.bin', size=100, verbose=True)
    #
    # model = word2vec.load(FLAGS.training_data_dir + 'word2vec_model.bin')
    # print('________________ Word2Vec model ready! ________________')
    # print('________________ Preparing the test data __________________')
    #
    # print('Nr. of test data stories: {0}'.format(len(test_stories)))
    #
    # test_highlights_file = open(FLAGS.test_data_dir + 'highlights.bin', 'w')
    # test_body_file = open(FLAGS.test_data_dir + 'body.bin', 'w')
    # test_stories_file = open(FLAGS.test_data_dir + 'stories.bin', 'w')
    # for s in test_stories:
    #     with open(FLAGS.cnn_stories_dir + s, 'rb') as story:
    #         r = story.read().decode('utf-8')
    #         highlights = [h.replace('\n\n', '|') for h in r.split('@highlight')[1:]]
    #         body = r.split('@highlight')[0]
    #         ann = nlp_client.annotate(body)
    #         sentences = [ann2sentence(s) + '\n' for s in ann.sentence]
    #         print(ann.sentence)
    #         if len(sentences) > 0:
    #             test_highlights_file.write(''.join(highlights) + '\n--\n')
    #             test_stories_file.write(s + '\n')
    #             test_body_file.write(''.join(sentences) + '\n--\n')
    #
    # print('________________ Preparing the val data __________________')
    # validation_highlights_file = open(FLAGS.validation_data_dir + 'highlights.bin', 'w')
    # validation_body_file = open(FLAGS.validation_data_dir + 'body.bin', 'w')
    # print('Nr. of val data stories: {0}'.format(len(val_stories)))
    # validation_stories_file = open(FLAGS.validation_data_dir + 'stories.bin', 'w')
    # for s in val_stories:
    #     with open(FLAGS.cnn_stories_dir + s, 'rb') as story:
    #         r = story.read().decode('utf-8')
    #         highlights = [h.replace('\n\n', '|') for h in r.split('@highlight')[1:]]
    #         body = r.split('@highlight')[0]
    #         ann = nlp_client.annotate(body)
    #         sentences = [ann2sentence(s) + '\n' for s in ann.sentence]
    #         if len(sentences) > 0:
    #             validation_highlights_file.write(''.join(highlights) + '\n--\n')
    #             validation_stories_file.write(s + '\n')
    #             validation_body_file.write(''.join(sentences) + '\n--\n')


if __name__ == '__main__':
    main()