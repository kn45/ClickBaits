#!/usr/bin/env python2.7
# -*- coding=utf-8 -*-

import cPickle
import fasttext
import logging
import numpy as np
import segjb
import sys
import xgboost as xgb

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s]: %(message)s")

w2v_file = 'xxxxx/fasttext_skipgram.model.bin'
w2v_model = None
model_file = 'gbt_model.pkl'
mdl_bst = None
hdl_seg = None


def title_encoding(title):
    smooth_factor = 2.
    title_u = str(title).decode('utf8')
    segs = hdl_seg.cut2list(title)
    vec_sent = np.zeros(200)
    word_cnt = 0.  # cnt of word collected in w2v model
    for seg in segs:
        vec_word = w2v_model[seg]
        vec_sent += vec_word
        word_cnt += 1
    if word_cnt > 0:
        vec_sent = vec_sent / (word_cnt + smooth_factor)
    return vec_sent


def pred():
    pred_feat_file = sys.argv[1]
    with open(pred_feat_file) as f:
        pred_data_raw = [l.rstrip('\n').split('\t') for l in f.readlines()]
    logging.info('prepare feature')
    pred_data_mod = [l[0] for l in pred_data_raw]
    pred_data_feat = np.array([title_encoding(l) for l in pred_data_mod])

    logging.info('start predicting')
    logging.info('best_iteration: ' + str(mdl_bst.best_iteration))
    pred_res = mdl_bst.predict(
        xgb.DMatrix(pred_data_feat),
        ntree_limit=mdl_bst.best_iteration)

    # output prediction result to stdout
    for misc, feats, res in zip(pred_data_raw, pred_data_feat, pred_res):
        pred = res
        if sum(feats**2) == 0.:
            pred = 0.
        print '\t'.join([str(pred)] + misc)


def init():
    logging.info('start init')

    # init seg
    global hdl_seg
    hdl_seg = segjb.SegJb()
    hdl_seg.init(user_dict='~/files/367w.dict.utf-8')
    hdl_seg.set_param(delim=' ', keep_stopwords=True, keep_puncs=False)

    # init fasttext-w2v
    global w2v_model
    w2v_model = fasttext.load_model(w2v_file)

    # init gbt
    global mdl_bst
    mdl_bst = cPickle.load(open(model_file, 'rb'))
    mdl_bst.set_param('nthread', 1)
    logging.info('finish init')


if __name__ == '__main__':
    init()
    pred()
