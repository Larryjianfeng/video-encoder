import numpy as np
import tensorflow as tf
from model import *
import json
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


labels = json.loads(open('../label.json').read())
l_dic = labels['data']
n = len(labels['label_0'])
n2 = len(labels['label_1'])
weights = [1.] * n

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parse_args():
    parser = argparse.ArgumentParser('arg parser for training video vectorize')
    parser.add_argument('--mode', type=str,
                        choices=['train', 'eval', 'infer'],
                        default='train')
    parser.add_argument('--method', type=str,
                        choices=['fvnet', 'netvlad', 'pooling'],
                        default='netvlad')
    parser.add_argument('--feature_size', type=int, default=300)
    parser.add_argument('--input_size', type=int, default=1536)
    parser.add_argument('--cluster_size', type=int, default=256)
    parser.add_argument('--use_length', type=bool, default=True)
    parser.add_argument('--vocab_size', type=int, default=-1)
    parser.add_argument('--vocab_size_2', type=int, default=-1)
    parser.add_argument('--use_2nd_label',  default=True,  type=boolean_string)
    parser.add_argument('--multitask_method', type=str, default='Attention')
    parser.add_argument('--is_training', default=True,  type=boolean_string)
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--global_max_len', type=int, default=30)
    parser.add_argument('--save_dir', type=str, default='./save.vlad/model.ckpt')
    parser.add_argument('--data_path', type=str, default='../train.inception.json')
    parser.add_argument('--gpu', type=str, default='0')

    return parser.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.vocab_size == -1:
    args.vocab_size = n
if args.vocab_size_2 == -1:
    args.vocab_size_2 = n2
print(args)

model = NetFV(feature_size=args.feature_size,
              cluster_size=args.cluster_size,
              use_length=args.use_length,
              vocab_size=args.vocab_size,
              input_size=args.input_size,
              vocab_size_2=args.vocab_size_2,
              use_2nd_label=args.use_2nd_label,
              multitask_method=args.multitask_method,
              method=args.method,
              is_training=args.is_training)

saver = tf.train.Saver(max_to_keep=1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


def proc(buff):
    inputs = []
    label = []
    label2 = []
    names = []
    for line in buff:
        try:
            k, v = list(line.items())[0]
            names.append(k)
            l = [0] * n
            l2 = [0] * n2
            w = [0.] * n
            for i, j in zip(*l_dic[k]):
                if i < n:
                    l[i] = 1
                    w[i] = j
                elif i < n + n2:
                    l2[i-n] = 1
            inputs.append(v)
            label.append(l)
            label2.append(l2)
        except Exception as e:
            print(e)
            continue
    max_len = min(max([len(li) for li in inputs]), args.global_max_len)
    length = [min(len(li), max_len) for li in inputs]
    inputs = [(li + [[0.] * len(inputs[0][0])] * (max_len - len(li)))[:max_len] for li in inputs]
    return (np.array(inputs, dtype=np.float32),
            np.array(length, dtype=np.int32),
            np.array(label, dtype=np.int32),
            np.array(label2, dtype=np.int32),
            names)


global_step = 0
try:
    saver.restore(sess, args.save_dir)
except Exception:
    print('train model from scratch')

for _ in range(args.num_iter):
    try:
        with open(args.data_path) as f:
            buff = []
            total_diff = [0, 0, 0]
            all_v, all_name = [], []
            while True:
                if len(buff) < args.batch_size:
                    buff.append(json.loads(f.readline()))
                else:
                    inputs, length, label, label2, names = proc(buff)
                    if args.mode == 'train':
                        _, _eval = (sess.run([model.train_op, model.eval_res], feed_dict={
                                             model.feeds: inputs,
                                             model.feeds_length: length,
                                             model.label: label,
                                             model.label_2: label2,
                                             model.weights: [weights]*len(label)}))
                        print(_eval)
                        if global_step % 50 == 0:
                            saver.save(sess, args.save_dir, global_step=None,
                                       write_meta_graph=False,
                                       write_state=False)

                        global_step += 1
                    elif args.mode == 'eval':
                        _eval = (sess.run(model.eval_res, feed_dict={
                                          model.feeds: inputs,
                                          model.feeds_length: length,
                                          model.label: label,
                                          model.label_2: label2,
                                          model.weights: [weights]*len(label)}))
                        total_diff[0] += _eval['avg_diff'] * len(label)
                        if args.use_2nd_label:
                            total_diff[1] += _eval['avg_diff2'] * len(label)
                        total_diff[2] += len(label)
                    elif args.mode == 'infer':
                        v = sess.run(model.repre, feed_dict={model.feeds: inputs, model.feeds_length: length,})
                        all_v.append(v)
                        all_name += names

                    buff = []

    except Exception as e:
        print(e)
        continue

if args.mode == 'eval':
    print(total_diff, total_diff[0]/(total_diff[2]+1e-9), total_diff[1]/(total_diff[2]+1e-9))
elif args.mode == 'infer':
    with open('repre.json', 'w') as f:
        all_v = np.concatenate(all_v, axis=0).tolist()
        f.write(json.dumps({'name': all_name, 'vec': all_v}))
