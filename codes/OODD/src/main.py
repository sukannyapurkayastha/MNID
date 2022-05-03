# python main.py -config ../config/config -section test-run
# cd src
# python main.py -config ../config/config -section test-run > ./log.txt
# cat ./log.txt | grep 'Meta-Test\|Meta-Valid' | python process_results.py

# ps -p PID
# ps -Flww -p PID
# kill PID


from __future__ import print_function
import numpy
import sys
import os
import configparser
from all_parameters import get_all_parameters
import tensorflow as tf
import argparse
from utils import compute_values, get_data
from experiment import RunExperiment
from workspace import workspace
from model import MatchingNetwork
from vocabulary import get_word_info
import math
import random
import json
import time
from tensorflow.python.client import device_lib
import pickle

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def train_test_model(params, sess, model, experiment, period):
    TEST_FILE_INDEX = 2
    train_data = get_data(params,
                          params['mnet_training_list'],
                          role='train_workspace',
                          period=period)
    
    dev_data = get_data(params,
                        params['mnet_dev_list'],
                        role='valid_workspace',
                        period=period)

    test_data = get_data(params,
                         params['testing_list'],
                         role='test_workspace',
                         period=period)

    best_v_macro_avg_eer = 1.0
    total_epoches = params['total_epoch']
    print('Starting epochs, Total =', total_epoches)
#     exit()
    for e in range(1, total_epoches+1):
        avg_loss = 0.0

        for workspace_idx in numpy.random.permutation(range(len(train_data))):
            curr_train_workspace = train_data[workspace_idx]
            all_ood_workapces = train_data[:workspace_idx] + train_data[workspace_idx+1:]
            batch_loss = experiment.run_training_epoch(curr_train_workspace, all_ood_workapces)
#             if params['single_workspace'] is False or e % 500 == 0:
#             if e%500 == 0:
#                 print("Epoch {}: train_loss: {}".format(e, batch_loss))
#             sys.stdout.flush()
            avg_loss += batch_loss
        avg_loss = avg_loss / len(train_data)
        print('Epoch {:4d}: avg_loss: {} {}'.format(e, avg_loss, os.getpid()))

#         if e % 50 == 0 or total_epoches==100:
        if 1==1:
            sys.stdout.flush()
            print('-'*40, 'Testing', '-'*40)
            v_macro_avg_eer, v_macro_avg_far, v_macro_avg_frr, \
                v_macro_avg_acc_ideal, v_macro_avg_acc, \
                val_output_info_list = compute_values(params, experiment, dev_data, e)

#             t_macro_avg_eer, t_macro_avg_far, t_macro_avg_frr, \
#                 t_macro_avg_acc_ideal, t_macro_avg_acc, \
#                 test_output_info_list = compute_values(params, experiment, test_data, e)

            print("Meta-Valid Macro(eer, onacc_ideal, onacc): %.3f, %.3f, %.3f" %
                  (v_macro_avg_eer, 
                   v_macro_avg_acc_ideal, 
                   v_macro_avg_acc))
#             print("Meta-Test Macro(eer, onacc_ideal, onacc): %.3f, %.3f, %.3f" %
#                   (t_macro_avg_eer,
#                    t_macro_avg_acc_ideal,
#                    t_macro_avg_acc))
            print('-'*89)

if __name__ == '__main__':
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser(description="Matching Network for few-shot learning.")
    parser.add_argument('-config', help="path to configuration file", 
                        default="../config/config")
    parser.add_argument('-section', help="the section name of the experiment",
                        default="test-run")
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--period', default=4, type=int)
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()
    config_paths = [args.config]
    config_parser = configparser.SafeConfigParser()
    config_found = config_parser.read(config_paths)

    params = get_all_parameters(config_parser, args.section)
    params['model_string'] = args.section
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    tf.set_random_seed(params['seed'])
    numpy.random.seed(params['seed'])
    random.seed(params['seed'])
    
    #################################
    for k in 'mnet_training_dir mnet_training_list mnet_dev_dir mnet_dev_list testing_dir testing_list'.split():
        params[k] = params[k].replace('data', f'DataProcessed/{args.dataset}')
    #################################

    print('Parameters:', json.dumps(params, indent=4))
    sys.stdout.flush()

    # build the vocabulary
    # if mode is train, vocabulary is train_data word + old word vectors
    # otherwise, vocabulary is read from wordd2idx file, params["wordvectors"]
    #            is initialized randomly, restored by model file later
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

#     print(get_available_gpus())
#     time.sleep(5)
    
    for chg in ['mnet_training_dir', 'mnet_training_list']:
        tmp = params[chg].split('/')
        tmp = tmp[:-1]+[f'Period{args.period}']+tmp[-1:]
        params[chg] = '/'.join(tmp)
        
    with tf.Session(config=config) as sess:
        voc, w2v = get_word_info(params)
        
        params['vocabulary'] = voc
        voclist = [None] * len(voc)
        for v in voc:
            voclist[voc[v]] = v
        params['voclist'] = voclist
        params["wordvectors"] = w2v
        
#     with open('params.txt', 'w') as outfile:
#         json.dumps(params, outfile)
    
#     with open('params.pkl', 'wb') as f:
#         pickle.dump(params, f)
        
    tf.reset_default_graph()
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():

            model = MatchingNetwork(params)
            model.init()
            experiment = RunExperiment(model, params, sess)
            sess.run(tf.global_variables_initializer())
            train_test_model(params, sess, model, experiment, args.period)
            print("=====Training Finished======")
