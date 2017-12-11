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
from tqdm import tqdm
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from threading import Lock
import mmap

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

class DataProcessor:
    def __init__(self, data_type, opt):
        self.data_type = data_type
        self.opt = opt
        self.source_path = os.path.join('data', self.data_type+'-v1.1.json')
        self.sink_path = os.path.join('data', 'processed_'+self.data_type+'-v1.1.json')
        self.glove_path = os.path.join('data', 'glove.'+self.opt['token_size']+'.'+str(self.opt['emb_dim'])+'d.txt')
        self.share_path = os.path.join('data', 'share_'+self.data_type+'.'+self.opt['token_size']+'.'+str(self.opt['emb_dim'])+'d.txt')
        self.batch_size = opt['batch_size']
        self.p_length = opt['p_length']
        self.q_length = opt['q_length']
        self.emb_dim = opt['emb_dim']
        self.queue_size = opt['queue_size']
        self.num_threads = opt['num_threads']
        self.read_batch = opt['read_batch']
        self.num_sample = 1
        self.no = 0
        self.no_lock = Lock()

    def process(self):
        '''
        pre-process the data
        1. tokenize all paragraphs, questions, answers
        2. find embedding for all words that showed up
        3. stored
        '''
        if os.path.isfile(self.sink_path):
            print('processed file already exists: {}'.format(self.sink_path))
            return
        with open(self.source_path, 'r') as source_file:
            source_data = json.load(source_file)
            sink_data = []

            n_article = len(source_data['data'])
            # memorize all words and create embedding efficiently
            word_map = set()
            articles = []
            print('Processing articles')
            for ai, article in enumerate(tqdm(source_data['data'])):
                paragraphs = []
                for pi, p in enumerate(article['paragraphs']):
                    context = p['context']
                    context_words = word_tokenize(context)
                    paragraphs.append(context_words)
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
                        answer_idx = [i + w_start for i in range(len(answer_words)) if i + w_start < num_words]
                        si, ei = answer_idx[0], answer_idx[-1]

                        sample = {
                            'ai': ai,
                            'pi': pi,
                            'question': question_words,
                            'answer': answer_words,
                            'si': si,
                            'ei': ei,
                            'id': qa['id']
                            }
                        sink_data.append(sample)
                articles.append(paragraphs[:]) 

        w2v = self.get_word_embedding(word_map)
        share_data = {
            'w2v': w2v,
            'articles': articles
        }
        print('Saving...')
        with open(self.share_path, 'w') as f:
            json.dump(share_data, f)
        with open(self.sink_path, 'w') as f:
            json.dump(sink_data, f)
        print('SQuAD '+self.data_type+' preprossing finished!')


    def get_word_embedding(self, word_map):
        print('generating embedding, this will take a while')
        word2vec = {}
        with open(self.glove_path, 'r', encoding='utf-8') as fn:
            for line in tqdm(fn, total=get_num_lines(self.glove_path)):
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
        print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec), len(word_map), self.glove_path))
        return word2vec

    def load_and_enqueue(self, sess, enqueue_op, coord, iden, debug):
        '''
        enqueues training sample, per read_batch per time 
        '''
        assert(self.sink_data)
        assert(self.share_data)
        print('feeder {} started'.format(iden))
        self.num_sample = len(self.sink_data)
        while not coord.should_stop():
            try:
                self.no_lock.acquire()
                start_idx = self.no % self.num_sample
                end_idx = (start_idx + self.read_batch) % self.num_sample
                self.no += self.read_batch
                self.no_lock.release()
                w2v_table = self.share_data['w2v']
                p = np.zeros((self.p_length, self.emb_dim))
                q = np.zeros((self.q_length, self.emb_dim))
                asi = np.zeros((1))
                aei = np.zeros((1))
                for i in range(start_idx, end_idx):
                    sample = self.sink_data[i]
                    question = sample['question']
                    paragraph = self.share_data['articles'][sample['ai']][sample['pi']]
                    for j in range(min(len(paragraph), self.p_length)):
                        try:
                            p[j] = w2v_table[paragraph[j]]
                        except KeyError:
                            pass
                    for j in range(min(len(question), self.q_length)):
                        try:
                            q[j] = w2v_table[question[j]]
                        except KeyError:
                            pass
                    asi[0] = sample['si']
                    aei[0] = sample['ei']
                    if self.data_type == 'dev':
                        paragraph_array = np.array([' '.join(paragraph)], dtype=object)
                    else:
                        paragraph_array = np.array([''], dtype=object)
                    # print('feeder {} before session run {}'.format(iden, i))
                    sess.run(enqueue_op, feed_dict={self.it['eP']: p, self.it['eQ']: q, self.it['asi']: asi, self.it['aei']: aei, self.it['p']: paragraph_array})
                    # print('enqueue operation runs')
            except:
                print('exception happens in feeder')
                coord.request_stop()
        

    def provide(self, sess):
        '''
        creates enqueue and dequeue operations to build input pipeline
        '''
        with open(self.sink_path, 'r') as data_raw, open(self.share_path, 'r') as share_raw:
            self.sink_data = json.load(data_raw)
            self.share_data = json.load(share_raw)
            self.num_sample = len(self.sink_data)
            # paragraph length filter: (train only)
            if self.data_type == 'train':
                self.sink_data = [sample for sample in self.sink_data if sample['ei'] < self.opt['p_length']]
        eP = tf.placeholder(tf.float32, [self.p_length, self.emb_dim])
        eQ = tf.placeholder(tf.float32, [self.q_length, self.emb_dim])
        asi = tf.placeholder(tf.int32, [1])
        aei = tf.placeholder(tf.int32, [1])
        p = tf.placeholder(tf.string, [1])
        self.it = {'eP': eP, 'eQ': eQ, 'asi': asi, 'aei': aei, 'p':p}
        with tf.variable_scope("queue"):
            self.q = tf.FIFOQueue(self.queue_size, [tf.float32, tf.float32, tf.int32, tf.int32, tf.string], shapes=[[self.p_length, self.emb_dim], [self.q_length, self.emb_dim], [1],[1],[1]])
            enqueue_op = self.q.enqueue([eP, eQ, asi, aei, p])
            # qr = tf.train.QueueRunner(q, [enqueue_op] * self.num_threads)
            # tf.train.add_queue_runner(qr)
            eP_batch, eQ_batch, asi_batch, aei_batch, p_batch = self.q.dequeue_many(self.batch_size)
            
        input_pipeline = {
            'eP': eP_batch,
            'eQ': eQ_batch,
            'asi': asi_batch,
            'aei': aei_batch,
            'p': p_batch
        }
        return input_pipeline, enqueue_op
        

def read_data(data_type, opt):
    return DataProcessor(data_type, opt)

def run():
    opt = json.load(open('models/config.json', 'r'))['rnet']
    dp_train = DataProcessor('train', opt)
    db_dev = DataProcessor('dev', opt)
    dp_train.process()
    db_dev.process()

if __name__ == "__main__":
	run()
