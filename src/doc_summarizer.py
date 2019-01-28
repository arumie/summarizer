from model_util import Model
from data_util import Data
from my_flags import FLAGS

def main():
    model = Model(feature_set=FLAGS['all_features'], type='NN')

if __name__ == '__main__':
    main()
