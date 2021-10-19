# -*- coding: utf-8 -*-

"""
Created on Wed Oct 21 22:49:38 2020

@author: 1952640
"""

import os
import sys
import json
import time
from io import TextIOWrapper
from datetime import timedelta
from pathlib import Path
import tensorflow as tf
from typing import *

from model import Model, TCNNConfig
from dataset import *

source_dir = Path(os.path.dirname(__file__))
root_dir = Path(os.path.dirname(source_dir))
data_dir = root_dir/'data'
train_path = data_dir/'train.txt'
test_path = data_dir/'test.txt'
val_path = data_dir/'val.txt'
vocab_path = data_dir/'vocab.txt'
save_dir = root_dir/'checkpoints'/'final'
save_path = save_dir/'best_validation'
tensorboard_dir = root_dir/'tensorboard'/'final'


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(viewed, candidates, keep_prob):
    return {
        model.input_click: viewed,
        model.input_candidate: candidates,
        model.keep_prob: keep_prob
    }


def evaluate(sess, x_, y_):
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, batch_size=config.batch_size,
                            max_length=config.num_words_title, candidate_num=config.candidate_len)
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for click, candidate, real in batch_eval:
        count += 1
        # print(candidate.shape)
        feed_dict = feed_data(click, candidate, real, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss
        total_acc += acc
    return total_loss / count, total_acc / count


def train():
    print("Configuring TensorBoard and Saver...")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    start_time = time.time()
    news_train, users_train = process_file(
        train_path, character_ids, category_ids, config.seq_length)
    news_val, users_val = process_file(
        val_path, character_ids, category_ids, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 200

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(
            news_train, users_train, batch_size=config.batch_size, max_length=config.num_words_title, candidate_num=config.candidate_len)
        acc = 0
        loss = 0
        for click, candidate, real in batch_train:
            feed_dict = feed_data(
                click, candidate, real, config.dropout_keep_prob)

            # print(x_batch.shape[0],x_batch.shape[1])
            if total_batch % config.save_per_batch == 0:
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            # and (total_batch!=0)):
            if ((total_batch % config.print_per_batch == 0)):
                feed_dict[model.keep_prob] = 1.0
                # loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                acc_train = acc/config.print_per_batch
                loss_train = loss/config.print_per_batch
                acc = 0
                loss = 0
                loss_val, acc_val = evaluate(
                    session, news_val, users_val)  # todo

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                    + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(
                    total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            feed_dict[model.keep_prob] = config.dropout_keep_prob
            # res_train = session.run(model.news_encoder.title_attention.attention_query_vector,feed_dict=feed_dict)
            # print(feed_dict)
            # print(res_train)
            # print(feed_dict)
            # session.run(model.optim, feed_dict=feed_dict)
            _loss, _acc, optim = session.run(
                [model.loss, model.acc, model.optim], feed_dict=feed_dict)
            loss += _loss
            acc += _acc
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    news_test, user_test, contents = test_process_file(
        test_path, character_ids, category_ids, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)
    batch_test = test_batch_iter(
        news_test, user_test, batch_size=config.batch_size, max_length=config.num_words_title, candidate_num=config.candidate_len)
    for click, candidate, real, nolist in batch_test:
        feed_dict = feed_data(click, candidate, real, 1.0)
        click_predict = session.run(
            model.click_probability, feed_dict=feed_dict)
        for i in range(1):
            # click 2 real 1 candidate 2
            print('\n user : ', user_test[nolist[i][2]])
            print(' click : ')
            for news in contents[nolist[i][0]:nolist[i][1]]:
                news_end = (30 if(len(news) >= 30) else len(news))
                print("".join('%s' % news[k] for k in range(0, news_end)))
            print(' candidate sort: ')
            print('score : %.2f' % (click_predict[i][0]*100))
            news = contents[nolist[i][2]]
            news_end = (30 if(len(news) >= 30) else len(news))
            print('content : ', "".join('%s' %
                                        news[k] for k in range(0, news_end)))
            for j in range(0, config.candidate_len-1):
                print('score : %.2f' % (click_predict[i][j+1]*100))
                news = contents[nolist[i][3]+j]
                news_end = (30 if(len(news) >= 30) else len(news))
                print('content : ', "".join('%s' %
                                            news[k] for k in range(0, news_end)))
        break
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def calc_confidence(data: str) -> list:
    newses: Dict[str, List[Dict[str, str]]] = json.loads(data)
    viewed_groups, candidates_groups = preprocess(
        newses['viewed'], newses['candidates'], character_ids, config.num_words_title)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, str(save_path))
    feed_dict = feed_data(viewed_groups, candidates_groups, 1.0)
    prediction: np.ndarray = session.run(
        model.confidence, feed_dict=feed_dict)
    return [value for group in prediction.tolist() for value in group][0:len(newses['candidates'])]


if __name__ == '__main__':
    sys.stdout = TextIOWrapper(sys.stdout.buffer, encoding='utf8')
    sys.stdin = TextIOWrapper(sys.stdin.buffer, encoding='utf8')
    config = TCNNConfig()
    config.batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if not os.path.exists(vocab_path):
        build_vocab(train_path, vocab_path, config.vocab_size)
    categories, category_ids = read_category()
    characters, character_ids = build_vocab(vocab_path)
    config.vocab_size = len(characters)
    model = Model(config)
    line: str
    for line in sys.stdin:
        args = line.strip().split(' ', 1)
        if args[0] == 'exit':
            break
        elif args[0] == 'run':
            print(calc_confidence(args[1]))
            sys.stdout.flush()