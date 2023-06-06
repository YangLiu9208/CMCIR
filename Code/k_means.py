import numpy as np
import json
import pickle
import torch
import math
import h5py
import random
from torch.utils.data import Dataset, DataLoader
from kmeans_pytorch import kmeans
 

if __name__ == '__main__':
    app= h5py.File('data/sutd-qa/sutd-qa_appearance_feat_swin_train_16.h5', 'r') 
    motion= h5py.File('data/sutd-qa/sutd-qa_motion_feat_swin_train_16.h5', 'r') 
    appearance_feat = []
    motion_feat = []

    for key in app['resnet_features']:
        appearance_feat.append(torch.from_numpy(key))  # (8, 16, 2048)
    appearance_feat = torch.stack(appearance_feat)
    appearance_feat=torch.mean(appearance_feat,(1,2))

    for key in motion['resnext_features']:
        motion_feat.append(torch.from_numpy(key))  # (8, 16, 2048)
    motion_feat = torch.stack(motion_feat)
    motion_feat = torch.mean(motion_feat,1)

    cluster_ids_app, cluster_centers_app = kmeans(X=appearance_feat, num_clusters=512, distance='euclidean', device=torch.device('cpu'))
    cluster_ids_motion, cluster_centers_motion = kmeans(X=motion_feat, num_clusters=512, distance='euclidean', device=torch.device('cpu'))

    app_out = h5py.File('data/sutd-qa/sutd-qa_appearance_swin16_feat_dict_512.h5', 'w')
    app_out.create_dataset('dict', data=cluster_centers_app)
    motion_out = h5py.File('data/sutd-qa/sutd-qa_motion_swin16_feat_dict_512.h5', 'w')
    motion_out.create_dataset('dict', data=cluster_centers_motion)