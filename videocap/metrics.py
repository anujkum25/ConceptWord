import argparse, os, pdb, sys, time
import numpy
import cPickle as pkl
import copy
import glob
import subprocess
from collections import OrderedDict
from cocoeval import COCOScorer
import pudb
MAXLEN = 80

def build_sample_pairs(samples, vidIDs):
    D = OrderedDict()
    for sample, vidID in zip(samples, vidIDs):
        D[vidID] = [{'image_id': vidID, 'caption': sample}]
    return D

def score_with_cocoeval(samples, ids):
    scorer = COCOScorer()
    if samples:
        gts = OrderedDict()
        for vidID in ids:
            gts[vidID] = engine.CAP[vidID]
        score = scorer.score(gts, samples, ids)
    else:
        score = None

    return score

def compute_score(sample, gts):
    #samples_valid = build_sample_pairs(sample_valid, engine.valid_ids)
    scorer = COCOScorer()
    #assert len(sample) == len(gts)
    ids = [str(x) for x in range(len(sample))]
    samp = OrderedDict()
    gts_dic = OrderedDict()
    for idx in ids:
        samp[idx] = [{'image_id': idx, 'caption':sample[int(idx)]}]
        gts_dic[idx] = [{'image_id': idx, 'cap_id':0, 'caption':gts[int(idx)], 'tokenized':gts[int(idx)]}]
    score = scorer.score(gts_dic, samp, ids)
    return score


def test_cocoeval():
    print 'Test here'

if __name__ == '__main__':
    test_cocoeval()
