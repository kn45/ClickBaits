# -*- coding=utf-8 -*-
import cPickle
import numpy as np
import os
import sys

"""Common tools for this project.
Utils are defined in this module for sharing.
"""

PROJ_DIR = os.path.split(os.path.realpath(__file__))[0]


def draw_progress(iteration, total, pref='Progress:', suff='',
                  decimals=1, barlen=50):
    """Call in a loop to create terminal progress bar
    """
    formatStr = "{0:." + str(decimals) + "f}"
    pcts = formatStr.format(100 * (iteration / float(total)))
    filledlen = int(round(barlen * iteration / float(total)))
    bar = '█' * filledlen + '-' * (barlen - filledlen)
    out_str = '\r%s |%s| %s%s %s' % (pref, bar, pcts, '%', suff)
    out_str = '\x1b[0;34;40m' + out_str + '\x1b[0m'
    sys.stderr.write(out_str),
    if iteration == total:
        sys.stderr.write('\n')
    sys.stderr.flush()


class CatEncoder(object):
    """Transform category to global uniq index
    """
    def __init__(self):
        self.cats = {}

    def build_dict(self, ifnames, columns):
        """need override
        ifnames are ',' separated
        fields are ',' separated, from 0. means from ... to
        """
        self.cats = {}
        cat_idx = 0
        ifnames = ifnames.split(',')
        cols = columns.split(',')
        col_st = int(cols[0])
        col_ed = int(cols[1]) if len(cols) > 1 else -1
        for ifname in ifnames:
            with open(ifname) as f:
                data = map(lambda l: l.strip('\n').split('\t'), f.readlines())
                for fields in data:
                    for idx in xrange(col_st, len(fields) if col_ed < 0 else
                                      col_ed+1):
                        if fields[idx] not in self.cats and fields[idx] != '':
                            self.cats[fields[idx]] = cat_idx
                            cat_idx += 1

    def save_dict(self, ofname):
        with open(ofname, 'w') as fo:
            for key in self.cats:
                print >> fo, \
                    '\t'.join([key.encode('utf8'), str(self.cats[key])])

    def load_dict(self, dfname):
        self.cats = {}
        with open(dfname) as f:
            data = [l.strip('\n').decode('utf8').split('\t')
                    for l in f.readlines()]
            for fields in data:
                self.cats[fields[0]] = int(fields[1])

    def n_cat(self):
        return len(self.cats)

    def cat2idx(self, cat):
        if cat in self.cats:
            return self.cats[cat]
        else:
            return -1

    def cat2onehot(self, cat, missing=False):
        idx = self.cat2idx(cat)
        if missing:
            res = [0] * (self.n_cat() + 1)
            idx = idx if idx >= 0 else (len(res) - 1)
            res[idx] = 1
            return res
        else:
            res = [0] * self.n_cat
            if idx > 0:
                res[idx] = 1
            return res


class CharEncoder(CatEncoder):
    def build_dict(self, ifname):
        """PAD: 0
        UNK: -1
        """
        self.cats = {}  # clean inner dict
        cat_idx = 1
        with open(ifname) as f:
            data = [x.strip('\n').split('\t')[1] for x in f.readlines()]
            for sent in data:
                for char in sent.decode('utf8'):
                    if char not in self.cats:
                        self.cats[char] = cat_idx
                        cat_idx += 1
        self.cats['UNK'] = cat_idx


def fill_missing_value(rec_fields):
    for idx, col in enumerate(rec_fields):
        if col == '':
            rec_fields[idx] = '-999.0'
    return rec_fields


if __name__ == '__main__':
    print "PROJ_DIR:\t" + PROJ_DIR
    from time import sleep
    for i in range(50):
        sleep(0.05)
        draw_progress(i, 49, pref='Progress:')
