#!/usr/bin/env python

import tensorflow as tf
import argparse
import json
import os
import string
from models import model_rnet
import preprocess

def run():
    parser = argparse.ArgumentParser(description='command line rnet trainer and evaluator')
    train_parser = parser.add_parser('train', help='train the network', dest='action')
    eval_parser = parser.add_parser('eval', help='evaluate the network', dest='action')
    #train_parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    train_parser.add_argument('--load', type=bool, default=False, help='load models')
    train_parser.add_argument('--save_dir', type=str, default='Models/save/', help='directory to save')

    args = parser.parse_args()

    if args['action'] == 'train':
        train(args)

    elif args['action'] == 'eval':
        evaluate(args)

def train(args):
    opt = json.load(open('models/config.json', 'r'))['rnet']['train']

    print('Reading data')
    db = preprocess.read_data('train', opt)
    data_input = db.create_input()

    rnet_model = model_rnet.R_NET(opt)
    loss, acc, pred_si, pred_ei = rnet_model.build_model()
    train_op = tf.train.AdadeltaOptimizer(1.0, rho=0.95, epsilon=1e-06).minimize(loss)

    startTime = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(500):
           _, loss_val, acc_val, p_si, p_ei = sess.run([train_op, loss, acc, pred_si, pred_ei])
           if i % 100 == 0:
               print('iter:{} - loss:{}'.format(i, loss_val))

        coord.request_stop()
        coord.join(threads)

    print('Training finished, took {} seconds'.format(time.time() - startTime))


def evaluate(args):


if __name__ == '__main__':
    run()
