#!/usr/bin/env python

import tensorflow as tf
import argparse
import json
import os
import string
import time
import threading
from tqdm import tqdm
import model
import preprocess

def run():
    parser = argparse.ArgumentParser(description='command line rnet trainer and evaluator')
    parser.add_argument('action', choices=['train', 'eval'])
    parser.add_argument('--load', type=bool, default=False, help='load models')
    parser.add_argument('--save_dir', type=str, default='Models/save/', help='directory to save')

    args = parser.parse_args()

    if args.action == 'train':
        train(args)

    elif args.action == 'eval':
        evaluate(args)

def feeder(dp, sess, enqueue_op, coord):
    dp.load_and_enqueue(sess, enqueue_op, coord)

def train(args):
    opt = json.load(open('models/config.json', 'r'))['rnet']

    print('Reading data')
    dp = preprocess.read_data('train', opt)
    sess = tf.Session()
    it, enqueue_op = dp.provide(sess)

    rnet_model = model.model.RNet(opt)
    loss, acc, pred_si, pred_ei = rnet_model.build_model(it)
    train_op = tf.train.AdadeltaOptimizer(1.0, rho=0.95, epsilon=1e-06).minimize(loss)

    startTime = time.time()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = []
        for i in range(opt['num_threads']):
            t = threading.Thread(target=feeder, args=(dp, sess, enqueue_op, coord))
            t.start()
            threads.append(t)
        
        for i in tqdm(range(5000)):
           _, loss_val, acc_val, p_si, p_ei = sess.run([train_op, loss, acc, pred_si, pred_ei])
           if i % 100 == 0:
               print('iter:{} - loss:{}'.format(i, loss_val))

        coord.request_stop()
        coord.join(threads)

    sess.close()
    print('Training finished, took {} seconds'.format(time.time() - startTime))


def evaluate(args):
    pass

if __name__ == '__main__':
    run()
