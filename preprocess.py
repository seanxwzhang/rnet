#!/usr/bin/env python

# coding=utf-8
import os
import re
import tensorflow as tf
from collections import Counter
import collections
import json
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

class DataProcessor:
    def __init__(self, data_type, opt):
        self.data_type = data_type
        self.opt = opt
        self.source_name = os.path.join('data', self.data_type+'-v1.1.json')
        self.sink_name = os.path.join('data', 'processed_'+self.data_type+'-v1.1.json')

    def process():
        '''
        pre-process the data
        1. tokenize all paragraphs, questions, answers
        2. find embedding for all words that showed up
        3. stored
        '''
        with open(self.source_name, 'r') as source_file:
            source_data = json.load(source_file)
            sink_data = []

            n_article = len(source_data['data'])
            # memorize all words and create embedding efficiently
            word_map = set()
            for ai, article in enumerate(source_data['data']):
                if ai%10 == 0:
                    print('processing article {}/{}'.format(ai, n_article)

                for pi, p in enumerate(article['paragraphs']):
                    context = p['context']
                    context_words = word_tokenize(context)
                    num_words = len(context_words)
                    for w in context_words:
                        word_map.add(w)

                    for qa in p['qas']:
                        question_words = word_tokenize(qa['question'])

                        # only care about the first answer
                        a = qa['answers'][0]
                        answer = a['text'].strip()
                        answer_start = int(a['answer_start'])
                        answer_words = word_tokenize(answer)

                        # need to find word level idx
                        w_start = len(word_tokenize(context[:answer_start]))
                        answer_idx = [i + w_start for i in xrange(answer_words) if i + w_start < num_words]
                        si, ei = answer_idx[0], answer_idx[-1]

                        sample = {
                            'ai': ai,
                            'pi': pi,
                            'question': question_words,
                            'answer': answer_words,
                            'si': si,
                            'ei': ei
                            'id': qa['id'],
                            }
                        data.append(sample)
            w2v, w2v_name = self.get_word_embedding(word_map)

        print('Saving...')
        with open(os.path.join('data', w2v_name), 'w') as f:
            json.dump(w2v, f)
        with open(os.path.join('data', self.sink_name), 'w') as f:
            json.dump(sink_data, f)

	print('SQuAD '+self.data_type+' preprossing finished!')


    def get_word_embedding(word_map):
        print('generating embedding')
        glove_path = os.path.join('data', 'glove.'+self.opt['token_size']+'.'+self.opt['glove_d']+'d.txt')
        word2vec = {}
        with open(glove_path, 'r', encoding='utf-8') as fn:
            for line in fn:
                array = line.strip().split(' ')
                w = array[0]
                v = list(map(float, array[1:]))
                if w in word_map:
                    word2vec[w] = v
                if w.capitalize() in word_map:
                    word2vec[w.capitalize()] = v
                if w.lower() in word_map:
                    word2vec[w.lower()] = v
                if w.upper() in word_map:
                    word2vec[w.upper()] = v
	print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec), len(word_map), glove_path))


    def provide():
        with tf.variable_scope('queue'):
            q = tf.FIFOQueue()
        data_path = os.path.join('data', 'data_{}.json'.format(data_type))
        self.data = self.load_data(data_path)
        pre_filter_length = len(self.data)

        # drop data if not conform to size criteria
        if self.data_type == 'train':
            self.data = [piece for piece in self.data if piece[]]

def read_data(data_type, opt):
    return DataProcessor(data_type, opt)

def run():
    train_data = DataProcessor('train', )

if __name__ == "__main__":
	run()
