# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import logging
import sys
import json
import scipy.io

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.evaluation import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer

cudnn.benchmark = True
logger = logging.getLogger('fastreid.visualize_result')

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        default=100,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="ascending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        default=10,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    logger = setup_logger()
    cfg = setup_cfg(args)
    test_loader, num_query = build_reid_test_loader(cfg, args.dataset_name)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    dataset = DATASET_REGISTRY.get(args.dataset_name)(root=_root)

    logger.info("Start extracting image features")
    feats = []
    pids = []
    camids = []
    for (feat, pid, camid) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)

    feats = torch.cat(feats, dim=0)
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])

    # compute cosine distance
    distmat = 1 - torch.mm(q_feat, g_feat.t())
    distmat = distmat.numpy()

    result = {'qg_fea': distmat}
    scipy.io.savemat('result.mat',result)

    # -----------------------split line-----------------------------------

    """ result = scipy.io.loadmat('result.mat')
    dastmat = result['qg_fea']
    
    result_dict = {}

    q_index = 2900  # query total number: 2900

    for i in range(q_index):

        index = distmat[i]
        index = np.argsort(distmat)  #from small to large
        index = index[::-1]

        query_path, _ =  dataset.query[i]
        #     query_path = '../train/pytorch/query/11/00002570.png'
        query_path = query_path.split('/')[-1] # get '00002570.png'

        img_path_list = []
        for j in range(200):                    # top-200
            img_path, _ = dataset.gallery[index[j]]
            #       img_path = '../train/pytorch/gallery/99/00108716.png'
            img_path = img_path.split('/')[-1]       # get '00108716.png'
            img_path_list.append(img_path)

        result_dict[query_path] = img_path_list
        if i % 500 == 0:
            print('{}/{} processed..........'.format(i+1, q_index))

    import datetime
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    with open('result_' + str(nowTime) + '.json','w') as fp:
        json.dump(result_dict, fp, indent = 4, separators=(',', ': '))
        # json.dump(result_dict, fp)

    print('The result generated................')

 """