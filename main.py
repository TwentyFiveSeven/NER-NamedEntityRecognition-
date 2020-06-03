#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
import argparse
import os
import datetime
import random
from model import Model
from dataset_batch import Dataset
from data_loader import data_loader
from evaluation import get_ner_bi_tag_list_in_sentence, diff_model_label, calculation_measure, calculation_measure_ensemble
from scipy import stats
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend" #작동하지 않는다 ..

def iteration_model(models, dataset, parameter, train=True):
    precision_count = np.zeros((parameter["num_ensemble"],2))
    recall_count = np.zeros((parameter["num_ensemble"],2))
    # 학습
    avg_cost = np.zeros(parameter["num_ensemble"])
    avg_correct = np.zeros(parameter["num_ensemble"])
    total_labels = np.zeros(parameter["num_ensemble"])
    correct_labels = np.zeros(parameter["num_ensemble"])
    dataset.shuffle_data()

    e_precision_count = np.array([ 0. , 0. ])
    e_recall_count = np.array([ 0. , 0. ])
    e_avg_correct = 0.0
    e_total_labels = 0.0
    if train:
        keep_prob = parameter["keep_prob"]
    else:
        keep_prob = 1.0
    count = 0
    step =0
    for morph, ne_dict, character, seq_len, char_len, label, step in dataset.get_data_batch_size(parameter["batch_size"], train):
        count +=1
        ensemble = []
        Fcount=0
        for i, model in enumerate(models):
            Fcount +=1
            #input_node 값을 할당하는 곳
            feed_dict = {model.morph: morph,
                         model.ne_dict: ne_dict,
                         model.character: character,
                         model.sequence: seq_len,
                         model.character_len: char_len,
                         model.label: label,
                         model.dropout_rate: keep_prob
                         }
            if train:
                cost, tf_viterbi_sequence, _ = sess.run([model.cost, model.viterbi_sequence, model.train_op], feed_dict=feed_dict)
            else:
                cost, tf_viterbi_sequence = sess.run([model.cost, model.viterbi_sequence], feed_dict=feed_dict)
            print('------------------* sess run end *------------------')
            ensemble.append(tf_viterbi_sequence)
            print(tf_viterbi_sequence)

            avg_cost[i] += cost

            mask = (np.expand_dims(np.arange(parameter["sentence_length"]), axis=0) <
                                np.expand_dims(seq_len, axis=1))
            total_labels[i] += np.sum(seq_len)

            correct_labels[i] = np.sum((label == tf_viterbi_sequence) * mask)
            avg_correct[i] += correct_labels[i]
            precision_count[i], recall_count[i] = diff_model_label(dataset, precision_count[i], recall_count[i], tf_viterbi_sequence, label, seq_len)


        ### ensemble

        ensemble = np.array(stats.mode(ensemble)[0][0])

        mask = (np.expand_dims(np.arange(parameter["sentence_length"]), axis=0) <
                np.expand_dims(seq_len, axis=1))
        e_total_labels += np.sum(seq_len)

        e_correct_labels = np.sum((label == ensemble) * mask)
        e_avg_correct += e_correct_labels
        e_precision_count, e_recall_count = diff_model_label(dataset, e_precision_count, e_recall_count,
                                                               ensemble, label, seq_len)

    return avg_cost / (step + 1), 100.0 * avg_correct / total_labels.astype(float), precision_count, recall_count, \
        100.0 * e_avg_correct / e_total_labels, e_precision_count, e_recall_count

if __name__ == '__main__':
    print('------------------* start *------------------')
    parser = argparse.ArgumentParser(description=sys.argv[0] + " description")
    parser.add_argument('--verbose', default=False, required=False, action='store_true', help='verbose')

    parser.add_argument('--mode', type=str, default="train", required=False, help='Choice operation mode')
    parser.add_argument('--iteration', type=int, default=0, help='fork 명령어를 사용할때 iteration 값에 매칭되는 모델이 로드됩니다.')
    parser.add_argument('--pause', type=int, default=0, help='모델이 load 될때 1로 설정됩니다.')

    parser.add_argument('--input_dir', type=str, default="data_in", required=False, help='Input data directory')
    parser.add_argument('--output_dir', type=str, default="data_out", required=False, help='Output data directory')
    parser.add_argument('--necessary_file', type=str, default="necessary.pkl")
    parser.add_argument('--train_lines', type=int, default=7457, required=False, help='Maximum train lines')
    # parser.add_argument('--train_lines', type=int, default=400, required=False, help='Maximum train lines')

    parser.add_argument('--epochs', type=int, default=2, required=False, help='Epoch value')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, required=False, help='Set learning rate')
    parser.add_argument('--keep_prob', type=float, default=0.65, required=False, help='Dropout_rate')

    parser.add_argument("--word_embedding_size", type=int, default=128, required=False, help='Word, WordPos Embedding Size')
    parser.add_argument("--char_embedding_size", type=int, default=128, required=False, help='Char Embedding Size')
    parser.add_argument("--tag_embedding_size", type=int, default=128, required=False, help='Tag Embedding Size')

    parser.add_argument('--lstm_units', type=int, default=128, required=False, help='Hidden unit size')
    parser.add_argument('--char_lstm_units', type=int, default=128, required=False, help='Hidden unit size for Char rnn')
    parser.add_argument('--sentence_length', type=int, default=6, required=False, help='Maximum words in sentence')
    parser.add_argument('--word_length', type=int, default=6, required=False, help='Maximum chars in word')
    parser.add_argument('--num_ensemble', type=int, default=5, required=False, help='Number of submodels')
    parser.add_argument('--file_append', type=bool, default=True, required=False, help='Update pickle')

    try:
        parameter = vars(parser.parse_args())
    except:
        parser.print_help()
        sys.exit(0)

    # data_loader를 이용해서 전체 데이터셋 가져옴
    # if nsml.HAS_DATASET:
    #     DATASET_PATH = nsml.DATASET_PATH
    # else:
    #     DATASET_PATH = 'data'
    DATASET_PATH = os.path.abspath("./data")

    # 가져온 문장별 데이터셋을 이용해서 각종 정보 및 학습셋 구성
    if parameter["mode"] == "train" and not os.path.exists(parameter["necessary_file"]):
        extern_data = data_loader(DATASET_PATH)
        Dataset(parameter, extern_data)
    elif parameter['file_append'] and os.path.exists(parameter["necessary_file"]):
        extern_data = data_loader(DATASET_PATH)
        Dataset(parameter, extern_data, True)


    test_data = []
    extern_data = []
    dataset = Dataset(parameter, extern_data)
    testset = Dataset(parameter, test_data)

    # Model 불러오기
    #모델을 만들 때 input_node, output_node name을 설정해줘야합니다.
    models = []
    for i in range(parameter["num_ensemble"]):
        models.append(Model(dataset.parameter, i))
        models[i].build_model()

    # tensorflow session 생성 및 초기화
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.train.Saver()

    # 학습
    if parameter["mode"] == "train":
        extern_data = data_loader(DATASET_PATH)
        random.shuffle(extern_data)
        print('size : '+str(len(extern_data)))
        test_data = extern_data[6600:]
        extern_data = extern_data[:6600]
        # test_data = extern_data[:240]
        # extern_data = extern_data[:300]
        testset.make_input_data(test_data)
        print(extern_data)
        dataset.make_input_data(extern_data)
        for epoch in range(parameter["epochs"]):
            print('epoch : '+str(epoch))
            avg_cost, avg_correct, precision_count, recall_count, e_avg_correct, e_precision_count, e_recall_count = iteration_model(models, dataset, parameter)
            for i in range(parameter["num_ensemble"]):
                print(str(i) + '_[Epoch: {:>0}] cost = {:>.6} Accuracy = {:>.6}'.format(epoch + 1, avg_cost[i], avg_correct[i]))
            print('Ensemble [Epoch: {:>4}]  Accuracy = {:>.6}'.format(epoch + 1, e_avg_correct))
            f1Measure, precision, recall = calculation_measure(parameter["num_ensemble"], precision_count, recall_count)
            e_f1Measure, e_precision, e_recall = calculation_measure_ensemble(e_precision_count, e_recall_count)
            for i in range(parameter["num_ensemble"]):
                print(str(i) + '_[Train] F1Measure : {:.6f} Precision : {:.6f} Recall : {:.6f}'.format(f1Measure[i], precision[i], recall[i]))
            print('Ensemble [Train] F1Measure : {:.6f} Precision : {:.6f} Recall : {:.6f}'.format(e_f1Measure, e_precision, e_recall))
            # nsml.report(summary=True, scope=locals(), train__loss=avg_cost, ac=avg_correct, F1=f1Measure, precision=precision, recall=recall, step=epoch)
            tf.train.Saver().save(sess, './sModel/trained.ckpt')
            tf.train.write_graph(sess.graph_def, "./sModel/", 'trained.pb', as_text=False)
            writer = tf.summary.FileWriter("./sModel/",sess.graph)
            tf.train.write_graph(sess.graph_def, "./sModel/", 'trained.txt', as_text=True)
            # datee = str(datetime.datetime.now())
            # tf.io.write_graph(
            #     sess.graph_def,
            #     "./sModel/",
            #     "tflite.pb",
            #     as_text=False
            # )
            # saver.save(sess,'./sModel/SaveModel')
            print('-'*100)
            avg_cost, avg_correct, precision_count, recall_count, e_avg_correct, e_precision_count, e_recall_count = iteration_model(models, testset, parameter, False)
            for i in range(parameter["num_ensemble"]):
                print(str(i) + '_Val : [Epoch: {:>4}] cost = {:>.6} Accuracy = {:>.6}'.format(epoch + 1, avg_cost[i], avg_correct[i]))
            print('Ensemble [Epoch: {:>0}]  Accuracy = {:>.6}'.format(epoch + 1, e_avg_correct))
            f1Measure, precision, recall = calculation_measure(parameter["num_ensemble"], precision_count, recall_count)
            e_f1Measure, e_precision, e_recall = calculation_measure_ensemble(e_precision_count, e_recall_count)
            for i in range(parameter["num_ensemble"]):
                print(str(i) + '_Val : [Val] F1Measure : {:.6f} Precision : {:.6f} Recall : {:.6f}'.format(f1Measure[i], precision[i], recall[i]))
            print('Ensemble [Val] F1Measure : {:.6f} Precision : {:.6f} Recall : {:.6f}'.format(e_f1Measure, e_precision,  e_recall))
            print('=' * 100)