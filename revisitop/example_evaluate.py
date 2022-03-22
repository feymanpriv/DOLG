# EXAMPLE_EVALUATE  Code to evaluate example results on ROxford and RParis datasets.
# Revisited protocol has 3 difficulty setups: Easy (E), Medium (M), and Hard (H), 
# and evaluates the performance using mean average precision (mAP), as well as mean precision @ k (mP@k)
#
# More details about the revisited annotation and evaluation can be found in:
# Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking, CVPR 2018
#
# Authors: Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., 2018
# Written by Feymanpriv 2021

import os
import os.path as osp
import io
import pickle

import numpy as np
from scipy.io import loadmat

from dataset import configdataset
from compute import compute_map

data_root =  osp.abspath(osp.dirname(osp.dirname(__file__))) 

test_dataset = 'roxford5k'
#test_dataset = 'rparis6k'  
print('>> {}: Evaluating test dataset...'.format(test_dataset)) 
GLOBAL_FEATURE_PATH='features/rparis6kdolgfea.pickle'
DISTRACTOR_FEATURE_PATH='features/1M.mat'
WITH_1M = False


def process(fea_pickle_path, cfg):
    features = {'Q':[], 'X':[]}
    with open(fea_pickle_path, "rb") as f:
        feadic = pickle.load(f)
        for qname in cfg['qimlist']:
            features['Q'].append(feadic[qname].reshape(-1))
        for name in cfg['imlist']:
            features['X'].append(feadic[name].reshape(-1))
    if WITH_1M:
        data = loadmat(DISTRACTOR_FEATURE_PATH)    
        features['X'] += list(data['X'].squeeze())
        print("distactors:", data['X'].squeeze().shape)

    features['Q'] = np.array(features['Q'])
    features['X'] = np.array(features['X'])
    print("final db:", features['X'].shape)
    return features
    
    
def global_search(features):
    """ rank by global descriptors """ 
    #features = loadmat(os.path.join(data_root, global_feature_path))
    Q = features['Q']
    X = features['X']

    sim = np.dot(X, Q.T)
    ranks = np.argsort(-sim, axis=0)
    #np.save("ranks_before_gv.npy", ranks)
    return ranks


def reportMAP(test_dataset, cfg, ranks):
    gnd = cfg['gnd']
    # evaluate ranks
    ks = [1, 5, 10]

    # search for easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

    print('>> {}: mAP E: {}, M: {}, H: {}'.format(test_dataset, 
          np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
    print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(test_dataset, np.array(ks), 
          np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))


def main():
    cfg = configdataset(test_dataset, data_root)
    features = process(GLOBAL_FEATURE_PATH, cfg)
    ranks = global_search(features) 
    reportMAP(test_dataset, cfg, ranks)
    print("Done!")


if __name__ == '__main__':
    main()
