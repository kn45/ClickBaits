#!/usr/bin/env python

import cPickle
import fasttext
import segjb
import numpy as np
import os
import sys
from mlfutil import *

data_file = sys.argv[1]

mfile = 'xxxx/fasttext_skipgram.model.bin'
model = None
hdl_seg = None


def title_encoding(title):
    smooth_factor = 2.
    title_u = str(title).decode('utf8')
    segs = hdl_seg.cut2list(title)
    vec_sent = np.zeros(200)
    word_cnt = 0.  # cnt of word collected in w2v model
    for seg in segs:
        vec_word = model[seg]
        vec_sent += vec_word
        word_cnt += 1
    if word_cnt > 0:
        vec_sent = vec_sent / (word_cnt + smooth_factor)
    return vec_sent


def init():
    # init_seg
    global hdl_seg
    hdl_seg = segjb.SegJb()
    hdl_seg.init(user_dict='~/files/367w.dict.utf-8')
    hdl_seg.set_param(delim=' ', keep_stopwords=True, keep_puncs=False)

    # init model
    global model
    model = fasttext.load_model(mfile)


def main():
    outfile = sys.argv[2]
    init()
    with open(data_file) as f:
        data = np.array([l.rstrip('\r\n').split('\t') for l in f.readlines()])
    fo = open(outfile, 'w')
    data_size = len(data)
    for nr, rec in enumerate(data):
        title = rec[1]
        title_feats = title_encoding(title).astype('|S10')
        print >> fo, '\t'.join(np.hstack([rec[0:1], title_feats]))
        draw_progress(nr, data_size-1)
    fo.close()

if __name__ == '__main__':
    main()
