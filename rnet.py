#!/usr/bin/env python

import tensorflow as tf
import argparse
import json
import os
import string
import time
import threading
from tqdm import tqdm
from models import model

import preprocess

def run():
    parser = argparse.ArgumentParser(description='command line rnet trainer and evaluator')
    parser.add_argument('action', choices=['train', 'eval'])
    parser.add_argument('--load', type=bool, default=False, help='load models')
    parser.add_argument('--epochs', type=int, default=1, help='Expochs')
    parser.add_argument('--save_dir', type=str, default='models/save/', help='directory to save')

    args = parser.parse_args()

    if args.action == 'train':
        train(args)

    elif args.action == 'eval':
        evaluate(args)

def feeder(dp, sess, enqueue_op, coord, i):
    dp.load_and_enqueue(sess, enqueue_op, coord, i)

def train(args):
    opt = json.load(open('models/config.json', 'r'))['rnet']
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    print('Reading data')
    dp = preprocess.read_data('train', opt)
    sess = tf.Session(config=config)
    it, enqueue_op = dp.provide(sess)

    rnet_model = model.RNet(opt)
    loss, pt = rnet_model.build_model(it)
    train_op = tf.train.AdadeltaOptimizer(1.0, rho=0.95, epsilon=1e-06).minimize(loss)

    # saving model
    saver = tf.train.Saver()

    startTime = time.time()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        # start feeding threads
        coord = tf.train.Coordinator()
        threads = []
        for i in range(opt['num_threads']):
            t = threading.Thread(target=feeder, args=(dp, sess, enqueue_op, coord, i))
            t.start()
            threads.append(t)
        # start training
        for i in range(args.epochs):
            print('Training...{}th epoch'.format(i))
            training_time = int(dp.num_sample/dp.batch_size)
            for i in tqdm(range(training_time)):
                _, loss_val, pt_val = sess.run([train_op, loss, pt])
                if i % 100 == 0:
                    print('iter:{} - loss:{}'.format(i, loss_val))
            save_path = saver.save(sess, os.path.join(args.save_dir, 'rnet_model{}.ckpt'.format(i)))
        
        coord.request_stop()
        coord.join(threads)
    
    save_path = saver.save(sess, os.path.join(args.save_dir, 'rnet_model_final_{}.ckpt'.format(time.strftime("%Y%m%d-%H%M%S"))))
    
    sess.close()
    print('Training finished, took {} seconds'.format(time.time() - startTime))

def evaluate(args):
    opt = json.load(open('models/config.json', 'r'))['rnet']
    print('Reading data')
    dp = preprocess.read_data('dev', opt)
    sess = tf.Session()
    it, enqueue_op = dp.provide(sess)
    loss, pt = model.build_model(it)

    saver = tf.train.Saver()



if __name__ == '__main__':
    run()
