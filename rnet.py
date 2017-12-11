#!/usr/bin/env python

import tensorflow as tf
import argparse
import json
import os
import string
import time
from threading import Thread, Lock
from tqdm import tqdm
from models import model
from evaluate import f1_score, exact_match_score
import preprocess

def run():
    parser = argparse.ArgumentParser(description='command line rnet trainer and evaluator')
    parser.add_argument('action', choices=['train', 'eval'])
    parser.add_argument('--load', type=bool, default=False, help='load models')
    parser.add_argument('--epochs', type=int, default=1, help='Expochs')
    parser.add_argument('--save_dir', type=str, default='models/save/', help='directory to save')
    parser.add_argument('--model_path', type=str, default='models/save/rnet_model_final.ckpt', help='saved model file')
    parser.add_argument('--debug', type=bool, default=False)

    args = parser.parse_args()

    if args.action == 'train':
        train(args)

    elif args.action == 'eval':
        evaluate(args)

def feeder(dp, sess, enqueue_op, coord, i, debug):
    dp.load_and_enqueue(sess, enqueue_op, coord, i, debug)

def train(args):
    opt = json.load(open('models/config.json', 'r'))['rnet']
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    print('Reading data')
    dp = preprocess.read_data('train', opt)
    sess = tf.Session(config=config)
    it, enqueue_op = dp.provide(sess)

    rnet_model = model.RNet(opt)
    loss, pt, accu = rnet_model.build_model(it)
    avg_loss = tf.reduce_mean(loss)
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
            t = Thread(target=feeder, args=(dp, sess, enqueue_op, coord, i, args.debug))
            t.start()
            threads.append(t)
        # start training
        for i in range(args.epochs):
            print('Training...{}th epoch'.format(i))
            training_time = int(dp.num_sample/dp.batch_size)
            for j in tqdm(range(training_time)):
                _, avg_loss_val, pt_val = sess.run([train_op, avg_loss, pt])
                if j % 100 == 0:
                    print('iter:{} - average loss:{}'.format(j, avg_loss_val))
            print('saving rnet_model{}.ckpt'.format(i))
            save_path = saver.save(sess, os.path.join(args.save_dir, 'rnet_model{}.ckpt'.format(i)))
        
        cancel_op = dp.q.close(cancel_pending_enqueues=True)
        sess.run(cancel_op)
        print('stopping feeders')
        coord.request_stop()
        coord.join(threads, ignore_live_threads=True)
    
    save_path = saver.save(sess, os.path.join(args.save_dir, 'rnet_model_final.ckpt'))
    
    sess.close()
    print('Training finished, took {} seconds'.format(time.time() - startTime))

def evaluate(args):
    opt = json.load(open('models/config.json', 'r'))['rnet']
    config = tf.ConfigProto(inter_op_parallelism_threads=1,intra_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saved_model = args.model_path
    
    EM = 0.0
    F1 = 0.0
    with sess.as_default():
        print('Reading data')
        dp = preprocess.read_data('dev', opt)
        it, enqueue_op = dp.provide(sess)
        rnet_model = model.RNet(opt)
        loss, pt, accu = rnet_model.build_model(it)
        dequeued_p, asi, aei = it['p'], it['asi'], it['aei']
        
         # restore model
        print('restoring model...')
        saver = tf.train.Saver()
        saver.restore(sess, saved_model)

        # start feeding threads
        coord = tf.train.Coordinator()

        threads = []
        for i in range(opt['num_threads']):
            t = Thread(target=feeder, args=(dp, sess, enqueue_op, coord, i, args.debug))
            t.start()
            threads.append(t)
        # start prediction
        print('Prediction starts')
        num_batch = int(dp.num_sample/dp.batch_size)
        for j in tqdm(range(num_batch)):
            pt_val, p_batch, asi_batch, aei_batch = sess.run([pt, dequeued_p, asi, aei])
            f1, em = 0.0, 0.0
            for k in range(len(p_batch)):
                paragraph = p_batch[k][0].decode('utf8').split(' ')
                true_start, true_end = asi_batch[k][0], aei_batch[k][0]
                pred_start, pred_end = pt_val[k][0], pt_val[k][1]
                pred_tokens = paragraph[pred_start:(pred_end+1)]
                true_tokens = paragraph[true_start:(true_end+1)]
                f1 += f1_score(' '.join(pred_tokens), ' '.join(true_tokens))
                em += exact_match_score(' '.join(pred_tokens), ' '.join(true_tokens))
            print('{}th batch | f1: {} | em: {}'.format(j, f1/len(p_batch), em/len(p_batch)))
            F1 += f1
            EM += em
        print('Evaluation complete, F1 score: {}, EM score: {}'.format(F1/dp.num_sample, EM/dp.num_sample))


if __name__ == '__main__':
    run()
