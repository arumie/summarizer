from model_util import NN
from my_flags import FLAGS

def main():
    neural_network = NN(feature_set=FLAGS['sent_embed'])

if __name__ == '__main__':
    main()