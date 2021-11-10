from __future__ import division

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from choiloader import ChoiDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import gensim
import utils
import os
import sys
from pathlib2 import Path
from wiki_loader import WikipediaDataSet
import accuracy
from models import naive
from timeit import default_timer as timer
import re
import math
from nltk.tokenize import RegexpTokenizer

missing_stop_words = set(['of', 'a', 'and', 'to'])
words_tokenizer = None

logger = utils.setup_logger(__name__, 'run_seg.log')

def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums

def getSegmentsFolders(path):

    ret_folders = []
    folders = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    for folder in folders:
        if folder.__contains__("-"):
            ret_folders.append(os.path.join(path,folder))
    return ret_folders

def get_words_tokenizer():
    global words_tokenizer

    if words_tokenizer:
        return words_tokenizer

    words_tokenizer = RegexpTokenizer(r'\w+')
    return words_tokenizer

def extract_sentence_words(sentence):
    tokenizer = get_words_tokenizer()
    sentence_words = tokenizer.tokenize(sentence)
    sentence_words = [w for w in sentence_words if w not in missing_stop_words]

    return sentence_words

def word_model(word, model):
    if model is None:
        return np.random.randn(1, 300)
    else:
        if word in model:
            return model[word].reshape(1, 300)
        else:
            return model['UNK'].reshape(1, 300)

def collate_fn(batch):
    batched_data = []

    window_size = 1
    before_sentence_count = int(math.ceil(float(window_size - 1) /2))
    after_sentence_count = window_size - before_sentence_count - 1

    for data in batch:
        try:
            max_index = len(data)
            tensored_data = []
            for curr_sentence_index in range(0, len(data)):
                from_index = max([0, curr_sentence_index - before_sentence_count])
                to_index = min([curr_sentence_index + after_sentence_count + 1, max_index])
                sentences_window = [word for sentence in data[from_index:to_index] for word in sentence]
                tensored_data.append(torch.FloatTensor(np.concatenate(sentences_window)))
            batched_data.append(tensored_data)
        except Exception as e:
            logger.debug('Exception!', exc_info=True)
            continue

    return batched_data


def split_encode_sentences(txt, word2vec):
    ### TODO: add more splitting operations
    sen_idxs = [m.start() for m in re.finditer(r'[\.\?\!]+', txt)]
    ini_idx = 0
    sentences = []

    for idx in sen_idxs:
        sen_txt = txt[ini_idx: idx+1].strip()
        ini_idx = idx + 1
        if len(sen_txt.strip()) > 0:
            # print("sentence: ", sen_txt)
            sentence_words = extract_sentence_words(sen_txt)
            if len(sentence_words) >= 1:
                sentences.append([word_model(word, word2vec) for word in sentence_words])
    # print(np.expand_dims(sentences, axis=0).shape)
    return collate_fn(np.expand_dims(sentences, axis=0))

def main(args):
    start = timer()
    ## load config
    utils.read_config_file(args.config)
    # print(utils.config)
    ## load word2vec
    word2vec = gensim.models.KeyedVectors.load(utils.config["word2vecfile"], mmap="r")
    word2vec_done = timer()
    print("loading word2vec: " + str(word2vec_done - start) + " s.")
    ## load model
    with open(args.model, "rb") as f:
        model = torch.load(f)
    model = maybe_cuda(model)
    model.eval()
    ## input sentences
    sen_encoded = split_encode_sentences(args.str, word2vec)
    # print("sen_encoded: ", len(sen_encoded[0]))
    output = model(sen_encoded)
    output_prob = softmax(output.data.cpu().numpy())
    output_seg = output_prob[:, 1] > args.seg_threshold
    # print("output_seg: ", output_seg)
    return output_seg


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', help='Model to run - will import and run (CPU only)', required=True)
    parser.add_argument('--str', help='input texts to be segmented', default = "Despite being a critical communication skill, grasping humor is challenging: a successful use of humor requires a mixture of both engaging content build-up and an appropriate vocal delivery (e.g., pause). Prior studies on computational humor emphasize the textual and audio features immediately next to the punchline, yet overlooking longer-term context setup. Moreover, the theories are usually too abstract for understanding each concrete humor snippet. To fill in the gap, we develop DeHumor, a visual analytical system for analyzing humorous behaviors in public speaking. To intuitively reveal the building blocks of each concrete example, DeHumor decomposes each humorous video into multimodal features and provides inline annotations of them on the video script. In particular, to better capture the build-ups, we introduce content repetition as a complement to features introduced in theories of computational humor and visualize them in a context linking graph. To help users locate the punchlines that have the desired features to learn, we summarize the content (with keywords) and humor feature statistics on an augmented time matrix. With case studies on stand-up comedy shows and TED talks, we show that DeHumor is able to highlight various building blocks of humor examples. In addition, expert interviews with communication coaches and humor researchers demonstrate the effectiveness of DeHumor for multimodal humor analysis of speech content and vocal delivery.")
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--seg_threshold', help='Threshold for binary classificetion', type=float, default=0.4)
    output_seg = main(parser.parse_args())
    print("output_seg: ", output_seg)



