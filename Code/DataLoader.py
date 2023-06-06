# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import numpy as np
import json
import pickle
import torch
import math
import h5py
import random
from torch.utils.data import Dataset, DataLoader
import skvideo.io
from PIL import Image
from preprocess.datautils import sutd_qa
from decord import VideoReader
from decord import cpu,gpu

def invert_dict(d):
    return {v: k for k, v in d.items()}

def extract_clips_with_consecutive_frames(path, num_clips=8, num_frames_per_clip=16):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        A list of raw features of clips.
    """
    clips = list()
    video_data = skvideo.io.vread(path)
    #video_data = VideoReader(path,ctx=cpu(0)) 
    total_frames = video_data.shape[0]
    img_size = (224, 224)
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1:num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        if clip_start < 0:
            clip_start = 0
        if clip_end > total_frames:
            clip_end = total_frames - 1
        clip = video_data[clip_start:clip_end]
        if clip_start == 0:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_start], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((added_frames, clip), axis=0)
        if clip_end == (total_frames - 1):
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_end], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((clip, added_frames), axis=0)
        new_clip = []
        for j in range(num_frames_per_clip):
            frame_data = clip[j]
            img = Image.fromarray(frame_data).resize(size=img_size)
            frame_data = np.asarray(img)
            frame_data = np.transpose(frame_data, (2, 0, 1))
            new_clip.append(frame_data)
        new_clip = np.asarray(new_clip)  # (num_frames, width, height, channels)
        clips.append(new_clip)
    return clips

def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab


class VideoQADataset(Dataset):

    def __init__(self, answers, ans_candidates, ans_candidates_len, questions, questions_len, video_ids, q_ids,
                 app_feature_h5, app_feat_id_to_index, motion_feature_h5, motion_feat_id_to_index):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.all_q_ids = q_ids
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index

        if not np.any(ans_candidates):
            self.question_type = 'openended'
        else:
            self.question_type = 'mulchoices'
            self.all_ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
            self.all_ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = torch.zeros(5)
        ans_candidates_len = torch.zeros(5)
        if self.question_type == 'mulchoices':
            ans_candidates = self.all_ans_candidates[index]
            ans_candidates_len = self.all_ans_candidates_len[index]
        question = self.all_questions[index]
        question_len = self.all_questions_len[index]
        video_idx = self.all_video_ids[index].item()
        question_idx = self.all_q_ids[index]
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]

        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['resnet_features'][app_index]  # (8, 16, 2048)

        with h5py.File(self.motion_feature_h5, 'r') as f_motion:
            motion_feat = f_motion['resnext_features'][motion_index]  # (8, 2048)

        appearance_feat = torch.from_numpy(appearance_feat)
        motion_feat = torch.from_numpy(motion_feat)

        return (
            video_idx, 
            question_idx, 
            answer, 
            ans_candidates, 
            ans_candidates_len, 
            appearance_feat, 
            motion_feat, 
            question,
            question_len)

    def __len__(self):
        return len(self.all_questions)


class VideoQADataset_oie(Dataset):

    def __init__(self, answers, ans_candidates, ans_candidates_len, questions, questions_subject, questions_relation, questions_object, \
        questions_len, questions_subject_len, questions_relation_len, questions_object_len, video_ids, q_ids, app_feature_h5, app_feat_id_to_index,motion_feature_h5, motion_feat_id_to_index, app_dict, motion_dict):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_subject = torch.LongTensor(np.asarray(questions_subject))
        self.all_questions_relation = torch.LongTensor(np.asarray(questions_relation))
        self.all_questions_object = torch.LongTensor(np.asarray(questions_object))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_questions_subject_len = torch.LongTensor(np.asarray(questions_subject_len))
        self.all_questions_relation_len = torch.LongTensor(np.asarray(questions_relation_len))
        self.all_questions_object_len = torch.LongTensor(np.asarray(questions_object_len))
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.all_q_ids = q_ids
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index
        self.app_dict = app_dict
        self.motion_dict = motion_dict

        if not np.any(ans_candidates):
            self.question_type = 'openended'
        else:
            self.question_type = 'mulchoices'
            self.all_ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
            self.all_ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = torch.zeros(5)
        ans_candidates_len = torch.zeros(5)
        if self.question_type == 'mulchoices':
            ans_candidates = self.all_ans_candidates[index]
            ans_candidates_len = self.all_ans_candidates_len[index]
        question = self.all_questions[index]
        question_subject = self.all_questions_subject[index]
        question_relation = self.all_questions_relation[index]
        question_object = self.all_questions_object[index]
        question_len = self.all_questions_len[index]
        question_subject_len = self.all_questions_subject_len[index]
        question_relation_len = self.all_questions_relation_len[index]
        question_object_len = self.all_questions_object_len[index]
        video_idx = self.all_video_ids[index].item()
        question_idx = self.all_q_ids[index]
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]

        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['resnet_features'][app_index]  # (8, 16, 2048)

        with h5py.File(self.motion_feature_h5, 'r') as f_motion:
            motion_feat = f_motion['resnext_features'][motion_index]  # (8, 2048)

        appearance_feat = torch.from_numpy(appearance_feat)
        motion_feat = torch.from_numpy(motion_feat)

        app= h5py.File(self.app_dict, 'r') 
        motion= h5py.File(self.motion_dict, 'r') 
    
        appearance_dict =[] 
        for key in app['dict']:
            appearance_dict.append(torch.from_numpy(key))  # (512, 1536)
        appearance_dict = torch.stack(appearance_dict)
        motion_dict=[]
        for key in motion['dict']:
            motion_dict.append(torch.from_numpy(key))  # (512, 1024)    
        motion_dict = torch.stack(motion_dict)

        return (
            video_idx, 
            question_idx, 
            answer, 
            ans_candidates, 
            ans_candidates_len, 
            appearance_feat, 
            motion_feat, 
            question,
            question_subject,
            question_relation,
            question_object,
            question_len,
            question_subject_len,
            question_relation_len,
            question_object_len,
            appearance_dict,
            motion_dict            
            )

    def __len__(self):
        return len(self.all_questions)

class VideoQADataset_Transformer(Dataset):

    def __init__(self, answers, ans_candidates, ans_candidates_len, questions, questions_subject, questions_relation, questions_object, \
        questions_len, questions_subject_len, questions_relation_len, questions_object_len, video_ids, q_ids, app_feature_h5, app_feat_id_to_index,motion_feature_h5, motion_feat_id_to_index):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_subject = torch.LongTensor(np.asarray(questions_subject))
        self.all_questions_relation = torch.LongTensor(np.asarray(questions_relation))
        self.all_questions_object = torch.LongTensor(np.asarray(questions_object))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_questions_subject_len = torch.LongTensor(np.asarray(questions_subject_len))
        self.all_questions_relation_len = torch.LongTensor(np.asarray(questions_relation_len))
        self.all_questions_object_len = torch.LongTensor(np.asarray(questions_object_len))
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.all_q_ids = q_ids
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index

        if not np.any(ans_candidates):
            self.question_type = 'openended'
        else:
            self.question_type = 'mulchoices'
            self.all_ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
            self.all_ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = torch.zeros(5)
        ans_candidates_len = torch.zeros(5)
        if self.question_type == 'mulchoices':
            ans_candidates = self.all_ans_candidates[index]
            ans_candidates_len = self.all_ans_candidates_len[index]
        question = self.all_questions[index]
        question_subject = self.all_questions_subject[index]
        question_relation = self.all_questions_relation[index]
        question_object = self.all_questions_object[index]
        question_len = self.all_questions_len[index]
        question_subject_len = self.all_questions_subject_len[index]
        question_relation_len = self.all_questions_relation_len[index]
        question_object_len = self.all_questions_object_len[index]
        video_idx = self.all_video_ids[index].item()
        question_idx = self.all_q_ids[index]
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]

        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['raw_appearance_clips'][app_index]  # (8, 16, 2048)

        with h5py.File(self.motion_feature_h5, 'r') as f_motion:
            motion_feat = f_motion['raw_motion_clips'][motion_index]  # (8, 2048)

        appearance_feat = torch.from_numpy(appearance_feat)
        motion_feat = torch.from_numpy(motion_feat)
        return (
            video_idx, 
            question_idx, 
            answer, 
            ans_candidates, 
            ans_candidates_len, 
            appearance_feat, 
            motion_feat, 
            question,
            question_subject,
            question_relation,
            question_object,
            question_len,
            question_subject_len,
            question_relation_len,
            question_object_len            
            )

    def __len__(self):
        return len(self.all_questions)


class VideoQADataset_oie_bert(Dataset):

    def __init__(self, answers, ans_candidates, ans_candidates_len, ans_bert_candidates,ans_bert_candidates_len,ans_bert_attention_mask,ans_candidates_token_type,questions, questions_subject, questions_relation, questions_object,question_input_bert,question_mask_bert,question_ids_bert,subject_input_bert,subject_mask_bert,subject_ids_bert,relation_input_bert,relation_mask_bert,relation_ids_bert,object_input_bert,object_mask_bert,object_ids_bert, \
        questions_bert_len,questions_subject_bert_len,questions_relation_bert_len,questions_object_bert_len,questions_len, questions_subject_len, questions_relation_len, questions_object_len, video_ids, q_ids, app_feature_h5, app_feat_id_to_index,motion_feature_h5, motion_feat_id_to_index, app_dict, motion_dict):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_subject = torch.LongTensor(np.asarray(questions_subject))
        self.all_questions_relation = torch.LongTensor(np.asarray(questions_relation))
        self.all_questions_object = torch.LongTensor(np.asarray(questions_object))

        self.all_questions_input_bert_len = questions_bert_len
        self.max_questions_input_bert_len = torch.tensor(max(questions_bert_len))
        self.all_subject_input_bert_len = questions_subject_bert_len
        self.max_subject_input_bert_len = torch.tensor(max(questions_subject_bert_len))
        self.all_relation_input_bert_len = questions_relation_bert_len
        self.max_relation_input_bert_len = torch.tensor(max(questions_relation_bert_len))
        self.all_object_input_bert_len = questions_object_bert_len
        self.max_object_input_bert_len = torch.tensor(max(questions_object_bert_len))
        self.all_questions_input_bert = question_input_bert
        self.all_questions_mask_bert = question_mask_bert
        self.all_questions_ids_bert = question_ids_bert
        self.all_subject_input_bert = subject_input_bert
        self.all_subject_mask_bert = subject_mask_bert
        self.all_subject_ids_bert = subject_ids_bert
        self.all_relation_input_bert = relation_input_bert
        self.all_relation_mask_bert = relation_mask_bert
        self.all_relation_ids_bert = relation_ids_bert
        self.all_object_input_bert = object_input_bert
        self.all_object_mask_bert = object_mask_bert
        self.all_object_ids_bert = object_ids_bert

        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_questions_subject_len = torch.LongTensor(np.asarray(questions_subject_len))
        self.all_questions_relation_len = torch.LongTensor(np.asarray(questions_relation_len))
        self.all_questions_object_len = torch.LongTensor(np.asarray(questions_object_len))
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.all_q_ids = q_ids
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index
        self.app_dict = app_dict
        self.motion_dict = motion_dict

        if not np.any(ans_candidates):
            self.question_type = 'openended'
        else:
            self.question_type = 'mulchoices'
            self.all_ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
            self.all_ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))
            self.all_ans_candidates_bert = torch.LongTensor(torch.from_numpy(np.asarray(ans_bert_candidates))).squeeze()
            self.all_ans_candidates_bert_len = torch.tensor(ans_bert_candidates_len).squeeze()
            self.all_ans_candidates_bert_attention_mask = torch.LongTensor(torch.from_numpy(np.asarray(ans_bert_attention_mask))).squeeze()
            self.all_ans_candidates_token_type = torch.LongTensor(torch.from_numpy(np.asarray(ans_candidates_token_type))).squeeze()


    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        #answer = torch.tensor(answer)
        ans_candidates = torch.zeros(5)
        ans_candidates_len = torch.zeros(5)
        if self.question_type == 'mulchoices':
            ans_candidates = self.all_ans_candidates[index]
            ans_candidates_len = self.all_ans_candidates_len[index]   
            ans_candidates_bert = self.all_ans_candidates_bert[index]
            ans_candidates_bert_len = self.all_ans_candidates_bert_len[index]
            ans_candidates_bert_attention_mask = self.all_ans_candidates_bert_attention_mask[index]
            ans_candidates_token_type = self.all_ans_candidates_token_type[index]

        question = self.all_questions[index]
        question_subject = self.all_questions_subject[index]
        question_relation = self.all_questions_relation[index]
        question_object = self.all_questions_object[index]

        question_input_bert_len=torch.tensor(self.all_questions_input_bert_len[index])
        subject_input_bert_len=torch.tensor(self.all_subject_input_bert_len[index])
        relation_input_bert_len=torch.tensor(self.all_relation_input_bert_len[index])
        object_input_bert_len=torch.tensor(self.all_object_input_bert_len[index])

        question_input_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_questions_input_bert[index]))).squeeze()
        question_mask_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_questions_mask_bert [index]))).squeeze()
        question_ids_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_questions_ids_bert[index]))).squeeze()
        subject_input_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_subject_input_bert[index]))).squeeze()
        subject_mask_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_subject_mask_bert[index]))).squeeze()
        subject_ids_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_subject_ids_bert[index]))).squeeze()
        relation_input_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_relation_input_bert [index]))).squeeze()
        relation_mask_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_relation_mask_bert[index]))).squeeze()
        relation_ids_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_relation_ids_bert[index]))).squeeze()
        object_input_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_object_input_bert[index]))).squeeze()
        object_mask_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_object_mask_bert[index]))).squeeze()
        object_ids_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_object_ids_bert[index]))).squeeze()

        if question_input_bert_len < self.max_questions_input_bert_len:
            zeros_pad=torch.zeros(self.max_questions_input_bert_len-question_input_bert_len)
            ones_pad=torch.ones(self.max_questions_input_bert_len-question_input_bert_len)
            question_input_bert = torch.cat((question_input_bert,zeros_pad),0)
            question_mask_bert = torch.cat((question_mask_bert,ones_pad),0)
            question_ids_bert = torch.cat((question_ids_bert,zeros_pad),0)
       
        if subject_input_bert_len < self.max_subject_input_bert_len:
            zeros_pad=torch.zeros(self.max_subject_input_bert_len-subject_input_bert_len)
            ones_pad=torch.ones(self.max_subject_input_bert_len-subject_input_bert_len)
            subject_input_bert = torch.cat((subject_input_bert,zeros_pad),0)
            subject_mask_bert = torch.cat((subject_mask_bert,ones_pad),0)
            subject_ids_bert = torch.cat((subject_ids_bert,zeros_pad),0)

        if relation_input_bert_len < self.max_relation_input_bert_len:
            zeros_pad=torch.zeros(self.max_relation_input_bert_len-relation_input_bert_len)
            ones_pad=torch.ones(self.max_relation_input_bert_len-relation_input_bert_len)
            relation_input_bert = torch.cat((relation_input_bert,zeros_pad),0)
            relation_mask_bert = torch.cat((relation_mask_bert,ones_pad),0)
            relation_ids_bert = torch.cat((relation_ids_bert,zeros_pad),0)

        if object_input_bert_len < self.max_object_input_bert_len:
            zeros_pad=torch.zeros(self.max_object_input_bert_len-object_input_bert_len)
            ones_pad=torch.ones(self.max_object_input_bert_len-object_input_bert_len)
            object_input_bert = torch.cat((object_input_bert,zeros_pad),0)
            object_mask_bert = torch.cat((object_mask_bert,ones_pad),0)
            object_ids_bert = torch.cat((object_ids_bert,zeros_pad),0)

        # question_bert_len = self.max_questions_input_bert_len
        # question_subject_bert_len = self.max_subject_input_bert_len
        # question_relation_bert_len = self.max_relation_input_bert_len
        # question_object_bert_len = self.max_object_input_bert_len

        question_len = self.all_questions_len[index]
        question_subject_len = self.all_questions_subject_len[index]
        question_relation_len = self.all_questions_relation_len[index]
        question_object_len = self.all_questions_object_len[index]
        video_idx = self.all_video_ids[index].item()
        question_idx = self.all_q_ids[index]
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]

        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['resnet_features'][app_index]  # (8, 16, 2048)

        with h5py.File(self.motion_feature_h5, 'r') as f_motion:
            motion_feat = f_motion['resnext_features'][motion_index]  # (8, 2048)

        appearance_feat = torch.from_numpy(appearance_feat)
        motion_feat = torch.from_numpy(motion_feat)

        app= h5py.File(self.app_dict, 'r') 
        motion= h5py.File(self.motion_dict, 'r') 
    
        appearance_dict =[] 
        for key in app['dict']:
            appearance_dict.append(torch.from_numpy(key))  # (512, 1536)
        appearance_dict = torch.stack(appearance_dict)
        motion_dict=[]
        for key in motion['dict']:
            motion_dict.append(torch.from_numpy(key))  # (512, 1024)    
        motion_dict = torch.stack(motion_dict)

        return (
            video_idx, 
            question_idx, 
            answer, 
            ans_candidates, 
            ans_candidates_len, 
            ans_candidates_bert,
            ans_candidates_bert_len,
            ans_candidates_bert_attention_mask,
            ans_candidates_token_type,
            appearance_feat, 
            motion_feat, 
            question_input_bert.int(),
            question_mask_bert.int(),
            question_ids_bert.int(),
            subject_input_bert.int(),
            subject_mask_bert.int(),
            subject_ids_bert.int(),
            relation_input_bert.int(),
            relation_mask_bert.int(),
            relation_ids_bert.int(),
            object_input_bert.int(),
            object_mask_bert.int(),
            object_ids_bert.int(),
            question_input_bert_len,
            subject_input_bert_len,
            relation_input_bert_len,
            object_input_bert_len,
            question,
            question_subject,
            question_relation,
            question_object,
            question_len,
            question_subject_len,
            question_relation_len,
            question_object_len,
            appearance_dict,
            motion_dict            
            )

    def __len__(self):
        return len(self.all_questions)

class VideoQADataset_oie_bert_openended(Dataset):

    def __init__(self, answers, ans_candidates, ans_candidates_len, questions, questions_subject, questions_relation, questions_object,question_input_bert,question_mask_bert,question_ids_bert,subject_input_bert,subject_mask_bert,subject_ids_bert,relation_input_bert,relation_mask_bert,relation_ids_bert,object_input_bert,object_mask_bert,object_ids_bert, \
        questions_bert_len,questions_subject_bert_len,questions_relation_bert_len,questions_object_bert_len,questions_len, questions_subject_len, questions_relation_len, questions_object_len, video_ids, q_ids, app_feature_h5, app_feat_id_to_index,motion_feature_h5, motion_feat_id_to_index, app_dict, motion_dict):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_subject = torch.LongTensor(np.asarray(questions_subject))
        self.all_questions_relation = torch.LongTensor(np.asarray(questions_relation))
        self.all_questions_object = torch.LongTensor(np.asarray(questions_object))

        self.all_questions_input_bert_len = questions_bert_len
        self.max_questions_input_bert_len = torch.tensor(max(questions_bert_len))
        self.all_subject_input_bert_len = questions_subject_bert_len
        self.max_subject_input_bert_len = torch.tensor(max(questions_subject_bert_len))
        self.all_relation_input_bert_len = questions_relation_bert_len
        self.max_relation_input_bert_len = torch.tensor(max(questions_relation_bert_len))
        self.all_object_input_bert_len = questions_object_bert_len
        self.max_object_input_bert_len = torch.tensor(max(questions_object_bert_len))
        self.all_questions_input_bert = question_input_bert
        self.all_questions_mask_bert = question_mask_bert
        self.all_questions_ids_bert = question_ids_bert
        self.all_subject_input_bert = subject_input_bert
        self.all_subject_mask_bert = subject_mask_bert
        self.all_subject_ids_bert = subject_ids_bert
        self.all_relation_input_bert = relation_input_bert
        self.all_relation_mask_bert = relation_mask_bert
        self.all_relation_ids_bert = relation_ids_bert
        self.all_object_input_bert = object_input_bert
        self.all_object_mask_bert = object_mask_bert
        self.all_object_ids_bert = object_ids_bert

        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_questions_subject_len = torch.LongTensor(np.asarray(questions_subject_len))
        self.all_questions_relation_len = torch.LongTensor(np.asarray(questions_relation_len))
        self.all_questions_object_len = torch.LongTensor(np.asarray(questions_object_len))
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.all_q_ids = q_ids
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index
        self.app_dict = app_dict
        self.motion_dict = motion_dict

        if not np.any(ans_candidates):
            self.question_type = 'openended'
        else:
            self.question_type = 'mulchoices'
            self.all_ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
            self.all_ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))
            # self.all_ans_candidates_bert = ans_bert_candidates
            # self.all_ans_candidates_bert_len = ans_bert_candidates_len
            # self.all_ans_candidates_bert_attention_mask = ans_bert_attention_mask
            # self.all_ans_candidates_token_type = ans_candidates_token_type


    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        #answer = torch.tensor(answer)
        ans_candidates = torch.zeros(5)
        ans_candidates_len = torch.zeros(5)
        if self.question_type == 'mulchoices':
            ans_candidates = self.all_ans_candidates[index]
            ans_candidates_len = self.all_ans_candidates_len[index]   
            # ans_candidates_bert = torch.LongTensor(self.all_ans_candidates_bert[index])
            # ans_candidates_bert_len = torch.tensor(self.all_ans_candidates_bert_len[index])
            # ans_candidates_bert_attention_mask = torch.LongTensor(torch.from_numpy(np.asarray(self.all_ans_candidates_bert_attention_mask[index]))).squeeze()
            # ans_candidates_token_type = torch.LongTensor(torch.from_numpy(np.asarray(self.all_ans_candidates_token_type[index]))).squeeze()

        question = self.all_questions[index]
        question_subject = self.all_questions_subject[index]
        question_relation = self.all_questions_relation[index]
        question_object = self.all_questions_object[index]

        question_input_bert_len=torch.tensor(self.all_questions_input_bert_len[index])
        subject_input_bert_len=torch.tensor(self.all_subject_input_bert_len[index])
        relation_input_bert_len=torch.tensor(self.all_relation_input_bert_len[index])
        object_input_bert_len=torch.tensor(self.all_object_input_bert_len[index])

        question_input_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_questions_input_bert[index]))).squeeze()
        question_mask_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_questions_mask_bert [index]))).squeeze()
        question_ids_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_questions_ids_bert[index]))).squeeze()
        subject_input_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_subject_input_bert[index]))).squeeze()
        subject_mask_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_subject_mask_bert[index]))).squeeze()
        subject_ids_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_subject_ids_bert[index]))).squeeze()
        relation_input_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_relation_input_bert [index]))).squeeze()
        relation_mask_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_relation_mask_bert[index]))).squeeze()
        relation_ids_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_relation_ids_bert[index]))).squeeze()
        object_input_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_object_input_bert[index]))).squeeze()
        object_mask_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_object_mask_bert[index]))).squeeze()
        object_ids_bert=torch.LongTensor(torch.from_numpy(np.asarray(self.all_object_ids_bert[index]))).squeeze()

        if question_input_bert_len < self.max_questions_input_bert_len:
            zeros_pad=torch.zeros(self.max_questions_input_bert_len-question_input_bert_len)
            ones_pad=torch.ones(self.max_questions_input_bert_len-question_input_bert_len)
            question_input_bert = torch.cat((question_input_bert,zeros_pad),0)
            question_mask_bert = torch.cat((question_mask_bert,ones_pad),0)
            question_ids_bert = torch.cat((question_ids_bert,zeros_pad),0)
       
        if subject_input_bert_len < self.max_subject_input_bert_len:
            zeros_pad=torch.zeros(self.max_subject_input_bert_len-subject_input_bert_len)
            ones_pad=torch.ones(self.max_subject_input_bert_len-subject_input_bert_len)
            subject_input_bert = torch.cat((subject_input_bert,zeros_pad),0)
            subject_mask_bert = torch.cat((subject_mask_bert,ones_pad),0)
            subject_ids_bert = torch.cat((subject_ids_bert,zeros_pad),0)

        if relation_input_bert_len < self.max_relation_input_bert_len:
            zeros_pad=torch.zeros(self.max_relation_input_bert_len-relation_input_bert_len)
            ones_pad=torch.ones(self.max_relation_input_bert_len-relation_input_bert_len)
            relation_input_bert = torch.cat((relation_input_bert,zeros_pad),0)
            relation_mask_bert = torch.cat((relation_mask_bert,ones_pad),0)
            relation_ids_bert = torch.cat((relation_ids_bert,zeros_pad),0)

        if object_input_bert_len < self.max_object_input_bert_len:
            zeros_pad=torch.zeros(self.max_object_input_bert_len-object_input_bert_len)
            ones_pad=torch.ones(self.max_object_input_bert_len-object_input_bert_len)
            object_input_bert = torch.cat((object_input_bert,zeros_pad),0)
            object_mask_bert = torch.cat((object_mask_bert,ones_pad),0)
            object_ids_bert = torch.cat((object_ids_bert,zeros_pad),0)

        # question_bert_len = self.max_questions_input_bert_len
        # question_subject_bert_len = self.max_subject_input_bert_len
        # question_relation_bert_len = self.max_relation_input_bert_len
        # question_object_bert_len = self.max_object_input_bert_len
        question_bert_len = question_input_bert_len
        question_subject_bert_len = subject_input_bert_len
        question_relation_bert_len = relation_input_bert_len
        question_object_bert_len = object_input_bert_len

        question_len = self.all_questions_len[index]
        question_subject_len = self.all_questions_subject_len[index]
        question_relation_len = self.all_questions_relation_len[index]
        question_object_len = self.all_questions_object_len[index]
        video_idx = self.all_video_ids[index].item()
        question_idx = self.all_q_ids[index]
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]

        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['resnet_features'][app_index]  # (8, 16, 2048)

        with h5py.File(self.motion_feature_h5, 'r') as f_motion:
            motion_feat = f_motion['resnext_features'][motion_index]  # (8, 2048)

        appearance_feat = torch.from_numpy(appearance_feat)
        motion_feat = torch.from_numpy(motion_feat)

        app= h5py.File(self.app_dict, 'r') 
        motion= h5py.File(self.motion_dict, 'r') 
    
        appearance_dict =[] 
        for key in app['dict']:
            appearance_dict.append(torch.from_numpy(key))  # (512, 1536)
        appearance_dict = torch.stack(appearance_dict)
        motion_dict=[]
        for key in motion['dict']:
            motion_dict.append(torch.from_numpy(key))  # (512, 1024)    
        motion_dict = torch.stack(motion_dict)

        return (
            video_idx, 
            question_idx, 
            answer, 
            ans_candidates, 
            ans_candidates_len, 
            # ans_candidates_bert,
            # ans_candidates_bert_len,
            # ans_candidates_bert_attention_mask,
            # ans_candidates_token_type,
            appearance_feat, 
            motion_feat, 
            question_input_bert.int(),
            question_mask_bert.int(),
            question_ids_bert.int(),
            subject_input_bert.int(),
            subject_mask_bert.int(),
            subject_ids_bert.int(),
            relation_input_bert.int(),
            relation_mask_bert.int(),
            relation_ids_bert.int(),
            object_input_bert.int(),
            object_mask_bert.int(),
            object_ids_bert.int(),
            question_bert_len,
            question_subject_bert_len,
            question_relation_bert_len,
            question_object_bert_len,
            question,
            question_subject,
            question_relation,
            question_object,
            question_len,
            question_subject_len,
            question_relation_len,
            question_object_len,
            appearance_dict,
            motion_dict            
            )

    def __len__(self):
        return len(self.all_questions)

class VideoQADataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')

        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        if 'train_num' in kwargs:
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                questions = questions[:trained_num]
                questions_len = questions_len[:trained_num]
                video_ids = video_ids[:trained_num]
                q_ids = q_ids[:trained_num]
                answers = answers[:trained_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:trained_num]
                    ans_candidates_len = ans_candidates_len[:trained_num]
        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                questions = questions[:val_num]
                questions_len = questions_len[:val_num]
                video_ids = video_ids[:val_num]
                q_ids = q_ids[:val_num]
                answers = answers[:val_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:val_num]
                    ans_candidates_len = ans_candidates_len[:val_num]
        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                questions = questions[:test_num]
                questions_len = questions_len[:test_num]
                video_ids = video_ids[:test_num]
                q_ids = q_ids[:test_num]
                answers = answers[:test_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:test_num]
                    ans_candidates_len = ans_candidates_len[:test_num]

        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}

        print('loading motion feature from %s' % (kwargs['motion_feat']))
        with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.motion_feature_h5 = kwargs.pop('motion_feat')
        self.dataset = VideoQADataset(answers, ans_candidates, ans_candidates_len, questions, questions_len,
                                      video_ids, q_ids,
                                      self.app_feature_h5, app_feat_id_to_index, self.motion_feature_h5,
                                      motion_feat_id_to_index)

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

class VideoQADataLoader_oie(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        vocab_subject_json_path = str(kwargs.pop('vocab_subject_json'))
        print('loading vocab_subject from %s' % (vocab_subject_json_path))
        vocab_subject = load_vocab(vocab_subject_json_path)

        vocab_relation_json_path = str(kwargs.pop('vocab_relation_json'))
        print('loading vocab_relation from %s' % (vocab_relation_json_path))
        vocab_relation = load_vocab(vocab_relation_json_path)

        vocab_object_json_path = str(kwargs.pop('vocab_object_json'))
        print('loading vocab_object from %s' % (vocab_object_json_path))
        vocab_object = load_vocab(vocab_object_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')

        question_subject_pt_path = str(kwargs.pop('question_subject_pt'))
        print('loading questions_subject from %s' % (question_subject_pt_path))

        question_relation_pt_path = str(kwargs.pop('question_relation_pt'))
        print('loading questions_relation from %s' % (question_relation_pt_path))

        question_object_pt_path = str(kwargs.pop('question_object_pt'))
        print('loading questions_object from %s' % (question_object_pt_path))

        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        with open(question_subject_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_subject = obj['questions']
            questions_subject_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_subject = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        with open(question_relation_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_relation = obj['questions']
            questions_relation_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_relation = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        with open(question_object_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_object = obj['questions']
            questions_object_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_object = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        if 'train_num' in kwargs:
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                questions = questions[:trained_num]
                questions_subject = questions_subject[:trained_num]
                questions_relation = questions_relation[:trained_num]
                questions_object = questions_object[:trained_num]
                questions_len = questions_len[:trained_num]
                questions_subject_len = questions_subject_len[:trained_num]
                questions_relation_len = questions_relation_len[:trained_num]
                questions_object_len = questions_object_len[:trained_num]                                                
                video_ids = video_ids[:trained_num]
                q_ids = q_ids[:trained_num]
                answers = answers[:trained_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:trained_num]
                    ans_candidates_len = ans_candidates_len[:trained_num]
        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                questions = questions[:val_num]
                questions_subject = questions_subject[:val_num]
                questions_relation = questions_relation[:val_num]
                questions_object = questions_object[:val_num]
                questions_len = questions_len[:val_num]
                questions_subject_len = questions_subject_len[:val_num]
                questions_relation_len = questions_relation_len[:val_num]
                questions_object_len = questions_object_len[:val_num]
                video_ids = video_ids[:val_num]
                q_ids = q_ids[:val_num]
                answers = answers[:val_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:val_num]
                    ans_candidates_len = ans_candidates_len[:val_num]
        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                questions = questions[:test_num]
                questions_subject = questions_subject[:test_num]
                questions_relation = questions_relation[:test_num]
                questions_object = questions_object[:test_num]
                questions_len = questions_len[:test_num]
                questions_subject_len = questions_subject_len[:test_num]
                questions_relation_len = questions_relation_len[:test_num]
                questions_object_len = questions_object_len[:test_num]
                video_ids = video_ids[:test_num]
                q_ids = q_ids[:test_num]
                answers = answers[:test_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:test_num]
                    ans_candidates_len = ans_candidates_len[:test_num]

        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}

        print('loading motion feature from %s' % (kwargs['motion_feat']))
        with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.motion_feature_h5 = kwargs.pop('motion_feat')
        self.app_dict_h5 = kwargs.pop('appearance_dict')
        self.motion_dict_h5 = kwargs.pop('motion_dict')
        self.dataset = VideoQADataset_oie(answers, ans_candidates, ans_candidates_len, questions, questions_subject, questions_relation, questions_object,\
             questions_len, questions_subject_len, questions_relation_len, questions_object_len, video_ids, q_ids, self.app_feature_h5, app_feat_id_to_index, self.motion_feature_h5, motion_feat_id_to_index, self.app_dict_h5, self.motion_dict_h5)

        self.vocab = vocab
        self.vocab_subject = vocab_subject
        self.vocab_relation = vocab_relation
        self.vocab_object = vocab_object

        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix
        self.glove_matrix_subject = glove_matrix_subject
        self.glove_matrix_relation = glove_matrix_relation
        self.glove_matrix_object = glove_matrix_object

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

class VideoQADataLoader_oie_mc(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        vocab_subject_json_path = str(kwargs.pop('vocab_subject_json'))
        print('loading vocab_subject from %s' % (vocab_subject_json_path))
        vocab_subject = load_vocab(vocab_subject_json_path)

        vocab_relation_json_path = str(kwargs.pop('vocab_relation_json'))
        print('loading vocab_relation from %s' % (vocab_relation_json_path))
        vocab_relation = load_vocab(vocab_relation_json_path)

        vocab_object_json_path = str(kwargs.pop('vocab_object_json'))
        print('loading vocab_object from %s' % (vocab_object_json_path))
        vocab_object = load_vocab(vocab_object_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')

        question_subject_pt_path = str(kwargs.pop('question_subject_pt'))
        print('loading questions_subject from %s' % (question_subject_pt_path))

        question_relation_pt_path = str(kwargs.pop('question_relation_pt'))
        print('loading questions_relation from %s' % (question_relation_pt_path))

        question_object_pt_path = str(kwargs.pop('question_object_pt'))
        print('loading questions_object from %s' % (question_object_pt_path))

        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        with open(question_subject_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_subject = obj['questions']
            questions_subject_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_subject = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        with open(question_relation_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_relation = obj['questions']
            questions_relation_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_relation = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        with open(question_object_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_object = obj['questions']
            questions_object_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_object = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        if 'train_num' in kwargs:
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                questions = questions[:trained_num]
                questions_subject = questions_subject[:trained_num]
                questions_relation = questions_relation[:trained_num]
                questions_object = questions_object[:trained_num]
                questions_len = questions_len[:trained_num]
                questions_subject_len = questions_subject_len[:trained_num]
                questions_relation_len = questions_relation_len[:trained_num]
                questions_object_len = questions_object_len[:trained_num]                                                
                video_ids = video_ids[:trained_num]
                q_ids = q_ids[:trained_num]
                answers = answers[:trained_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:trained_num]
                    ans_candidates_len = ans_candidates_len[:trained_num]
        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                questions = questions[:val_num]
                questions_subject = questions_subject[:val_num]
                questions_relation = questions_relation[:val_num]
                questions_object = questions_object[:val_num]
                questions_len = questions_len[:val_num]
                questions_subject_len = questions_subject_len[:val_num]
                questions_relation_len = questions_relation_len[:val_num]
                questions_object_len = questions_object_len[:val_num]
                video_ids = video_ids[:val_num]
                q_ids = q_ids[:val_num]
                answers = answers[:val_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:val_num]
                    ans_candidates_len = ans_candidates_len[:val_num]
        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                questions = questions[:test_num]
                questions_subject = questions_subject[:test_num]
                questions_relation = questions_relation[:test_num]
                questions_object = questions_object[:test_num]
                questions_len = questions_len[:test_num]
                questions_subject_len = questions_subject_len[:test_num]
                questions_relation_len = questions_relation_len[:test_num]
                questions_object_len = questions_object_len[:test_num]
                video_ids = video_ids[:test_num]
                q_ids = q_ids[:test_num]
                answers = answers[:test_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:test_num]
                    ans_candidates_len = ans_candidates_len[:test_num]

        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}

        print('loading motion feature from %s' % (kwargs['motion_feat']))
        with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.motion_feature_h5 = kwargs.pop('motion_feat')
        self.app_dict_h5 = kwargs.pop('appearance_dict')
        self.motion_dict_h5 = kwargs.pop('motion_dict')
        self.dataset = VideoQADataset_oie(answers, ans_candidates, ans_candidates_len, questions, questions_subject, questions_relation, questions_object,\
             questions_len, questions_subject_len, questions_relation_len, questions_object_len, video_ids, q_ids, self.app_feature_h5, app_feat_id_to_index, self.motion_feature_h5, motion_feat_id_to_index, self.app_dict_h5, self.motion_dict_h5)

        self.vocab = vocab
        self.vocab_subject = vocab_subject
        self.vocab_relation = vocab_relation
        self.vocab_object = vocab_object

        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix
        self.glove_matrix_subject = glove_matrix_subject
        self.glove_matrix_relation = glove_matrix_relation
        self.glove_matrix_object = glove_matrix_object

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

class VideoQADataLoader_Transformer(DataLoader):

    def __init__(self, **kwargs):

        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        vocab_subject_json_path = str(kwargs.pop('vocab_subject_json'))
        print('loading vocab_subject from %s' % (vocab_subject_json_path))
        vocab_subject = load_vocab(vocab_subject_json_path)

        vocab_relation_json_path = str(kwargs.pop('vocab_relation_json'))
        print('loading vocab_relation from %s' % (vocab_relation_json_path))
        vocab_relation = load_vocab(vocab_relation_json_path)

        vocab_object_json_path = str(kwargs.pop('vocab_object_json'))
        print('loading vocab_object from %s' % (vocab_object_json_path))
        vocab_object = load_vocab(vocab_object_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')

        question_subject_pt_path = str(kwargs.pop('question_subject_pt'))
        print('loading questions_subject from %s' % (question_subject_pt_path))

        question_relation_pt_path = str(kwargs.pop('question_relation_pt'))
        print('loading questions_relation from %s' % (question_relation_pt_path))

        question_object_pt_path = str(kwargs.pop('question_object_pt'))
        print('loading questions_object from %s' % (question_object_pt_path))

        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        with open(question_subject_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_subject = obj['questions']
            questions_subject_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_subject = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        with open(question_relation_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_relation = obj['questions']
            questions_relation_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_relation = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        with open(question_object_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_object = obj['questions']
            questions_object_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_object = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']

        if 'train_num' in kwargs:
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                questions = questions[:trained_num]
                questions_subject = questions_subject[:trained_num]
                questions_relation = questions_relation[:trained_num]
                questions_object = questions_object[:trained_num]
                questions_len = questions_len[:trained_num]
                questions_subject_len = questions_subject_len[:trained_num]
                questions_relation_len = questions_relation_len[:trained_num]
                questions_object_len = questions_object_len[:trained_num]                                                
                video_ids = video_ids[:trained_num]
                q_ids = q_ids[:trained_num]
                answers = answers[:trained_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:trained_num]
                    ans_candidates_len = ans_candidates_len[:trained_num]
        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                questions = questions[:val_num]
                questions_subject = questions_subject[:val_num]
                questions_relation = questions_relation[:val_num]
                questions_object = questions_object[:val_num]
                questions_len = questions_len[:val_num]
                questions_subject_len = questions_subject_len[:val_num]
                questions_relation_len = questions_relation_len[:val_num]
                questions_object_len = questions_object_len[:val_num]
                video_ids = video_ids[:val_num]
                q_ids = q_ids[:val_num]
                answers = answers[:val_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:val_num]
                    ans_candidates_len = ans_candidates_len[:val_num]
        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                questions = questions[:test_num]
                questions_subject = questions_subject[:test_num]
                questions_relation = questions_relation[:test_num]
                questions_object = questions_object[:test_num]
                questions_len = questions_len[:test_num]
                questions_subject_len = questions_subject_len[:test_num]
                questions_relation_len = questions_relation_len[:test_num]
                questions_object_len = questions_object_len[:test_num]
                video_ids = video_ids[:test_num]
                q_ids = q_ids[:test_num]
                answers = answers[:test_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:test_num]
                    ans_candidates_len = ans_candidates_len[:test_num]

        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}

        print('loading motion feature from %s' % (kwargs['motion_feat']))
        with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.motion_feature_h5 = kwargs.pop('motion_feat')
        self.dataset = VideoQADataset_Transformer(answers, ans_candidates, ans_candidates_len, questions, questions_subject, questions_relation, questions_object,\
             questions_len, questions_subject_len, questions_relation_len, questions_object_len, video_ids, q_ids, self.app_feature_h5, app_feat_id_to_index, self.motion_feature_h5, motion_feat_id_to_index)

        self.vocab = vocab
        self.vocab_subject = vocab_subject
        self.vocab_relation = vocab_relation
        self.vocab_object = vocab_object

        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix
        self.glove_matrix_subject = glove_matrix_subject
        self.glove_matrix_relation = glove_matrix_relation
        self.glove_matrix_object = glove_matrix_object

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

class VideoQADataLoader_oie_bert(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        vocab_subject_json_path = str(kwargs.pop('vocab_subject_json'))
        print('loading vocab_subject from %s' % (vocab_subject_json_path))
        vocab_subject = load_vocab(vocab_subject_json_path)

        vocab_relation_json_path = str(kwargs.pop('vocab_relation_json'))
        print('loading vocab_relation from %s' % (vocab_relation_json_path))
        vocab_relation = load_vocab(vocab_relation_json_path)

        vocab_object_json_path = str(kwargs.pop('vocab_object_json'))
        print('loading vocab_object from %s' % (vocab_object_json_path))
        vocab_object = load_vocab(vocab_object_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')

        question_subject_pt_path = str(kwargs.pop('question_subject_pt'))
        print('loading questions_subject from %s' % (question_subject_pt_path))

        question_relation_pt_path = str(kwargs.pop('question_relation_pt'))
        print('loading questions_relation from %s' % (question_relation_pt_path))

        question_object_pt_path = str(kwargs.pop('question_object_pt'))
        print('loading questions_object from %s' % (question_object_pt_path))

        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            question_input_bert = obj['question_input_bert']
            question_mask_bert = obj['question_mask_bert']
            question_ids_bert = obj['question_ids_bert']
            questions_bert_len = obj['question_bert_len']
            questions_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            ans_bert_candidates = obj['ans_bert_candidates']
            ans_bert_candidates_len = obj['ans_candidates_bert_len']
            ans_bert_attention_mask = obj['ans_bert_attention_mask']
            ans_candidates_token_type = obj['ans_candidates_token_type']
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']
                ans_bert_candidates = obj['ans_bert_candidates']
                ans_bert_candidates_len = obj['ans_candidates_bert_len']
                ans_bert_attention_mask = obj['ans_bert_attention_mask']
                ans_candidates_token_type = obj['ans_candidates_token_type']

        with open(question_subject_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_subject = obj['questions']
            subject_input_bert = obj['question_input_bert']
            subject_mask_bert = obj['question_mask_bert']
            subject_ids_bert = obj['question_ids_bert']
            questions_subject_bert_len = obj['question_bert_len']
            questions_subject_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_subject = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            ans_bert_candidates = obj['ans_bert_candidates']
            ans_bert_candidates_len = obj['ans_candidates_bert_len']
            ans_bert_attention_mask = obj['ans_bert_attention_mask']
            ans_candidates_token_type = obj['ans_candidates_token_type']
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']
                ans_bert_candidates = obj['ans_bert_candidates']
                ans_bert_candidates_len = obj['ans_candidates_bert_len']
                ans_bert_attention_mask = obj['ans_bert_attention_mask']
                ans_candidates_token_type = obj['ans_candidates_token_type']

        with open(question_relation_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_relation = obj['questions']
            relation_input_bert = obj['question_input_bert']
            relation_mask_bert = obj['question_mask_bert']
            relation_ids_bert = obj['question_ids_bert']
            questions_relation_bert_len = obj['question_bert_len']
            questions_relation_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_relation = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            ans_bert_candidates = obj['ans_bert_candidates']
            ans_bert_candidates_len = obj['ans_candidates_bert_len']
            ans_bert_attention_mask = obj['ans_bert_attention_mask']
            ans_candidates_token_type = obj['ans_candidates_token_type']
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']
                ans_bert_candidates = obj['ans_bert_candidates']
                ans_bert_candidates_len = obj['ans_candidates_bert_len']
                ans_bert_attention_mask = obj['ans_bert_attention_mask']
                ans_candidates_token_type = obj['ans_candidates_token_type']

        with open(question_object_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_object = obj['questions']
            object_input_bert = obj['question_input_bert']
            object_mask_bert = obj['question_mask_bert']
            object_ids_bert = obj['question_ids_bert']
            questions_object_bert_len = obj['question_bert_len']
            questions_object_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_object = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            ans_bert_candidates = obj['ans_bert_candidates']
            ans_bert_candidates_len = obj['ans_candidates_bert_len']
            ans_bert_attention_mask = obj['ans_bert_attention_mask']
            ans_candidates_token_type = obj['ans_candidates_token_type']
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']
                ans_bert_candidates = obj['ans_bert_candidates']
                ans_bert_candidates_len = obj['ans_candidates_bert_len']
                ans_bert_attention_mask = obj['ans_bert_attention_mask']
                ans_candidates_token_type = obj['ans_candidates_token_type']

        if 'train_num' in kwargs:
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                questions = questions[:trained_num]
                questions_subject = questions_subject[:trained_num]
                questions_relation = questions_relation[:trained_num]
                questions_object = questions_object[:trained_num]

                question_input_bert = question_input_bert[:trained_num]
                question_mask_bert = question_mask_bert[:trained_num]
                question_ids_bert = question_ids_bert[:trained_num]

                subject_input_bert = subject_input_bert[:trained_num]
                subject_mask_bert = subject_mask_bert[:trained_num]
                subject_ids_bert = subject_ids_bert[:trained_num]

                relation_input_bert = relation_input_bert[:trained_num]
                relation_mask_bert = relation_mask_bert[:trained_num]
                relation_ids_bert = relation_ids_bert[:trained_num]

                object_input_bert = object_input_bert[:trained_num]
                object_mask_bert = object_mask_bert[:trained_num]
                object_ids_bert = object_ids_bert[:trained_num]

                questions_bert_len = questions_bert_len[:trained_num]
                questions_subject_bert_len = questions_subject_bert_len[:trained_num]
                questions_relation_bert_len = questions_relation_bert_len[:trained_num]
                questions_object_bert_len = questions_object_bert_len[:trained_num]  

                questions_len = questions_len[:trained_num]
                questions_subject_len = questions_subject_len[:trained_num]
                questions_relation_len = questions_relation_len[:trained_num]
                questions_object_len = questions_object_len[:trained_num]                                                
                video_ids = video_ids[:trained_num]
                q_ids = q_ids[:trained_num]
                answers = answers[:trained_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:trained_num]
                    ans_candidates_len = ans_candidates_len[:trained_num]
                    ans_bert_candidates = ans_bert_candidates[:trained_num]
                    ans_bert_candidates_len = ans_bert_candidates_len[:trained_num]
                    ans_bert_attention_mask = ans_bert_attention_mask[:trained_num]
                    ans_candidates_token_type = ans_candidates_token_type[:trained_num]
        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                questions = questions[:val_num]
                questions_subject = questions_subject[:val_num]
                questions_relation = questions_relation[:val_num]
                questions_object = questions_object[:val_num]

                question_input_bert = question_input_bert[:val_num]
                question_mask_bert = question_mask_bert[:val_num]
                question_ids_bert = question_ids_bert[:val_num]

                subject_input_bert = subject_input_bert[:val_num]
                subject_mask_bert = subject_mask_bert[:val_num]
                subject_ids_bert = subject_ids_bert[:val_num]

                relation_input_bert = relation_input_bert[:val_num]
                relation_mask_bert = relation_mask_bert[:val_num]
                relation_ids_bert = relation_ids_bert[:val_num]

                object_input_bert = object_input_bert[:val_num]
                object_mask_bert = object_mask_bert[:val_num]
                object_ids_bert = object_ids_bert[:val_num]

                questions_bert_len = questions_bert_len[:val_num]
                questions_subject_bert_len = questions_subject_bert_len[:val_num]
                questions_relation_bert_len = questions_relation_bert_len[:val_num]
                questions_object_bert_len = questions_object_bert_len[:val_num]  

                questions_len = questions_len[:val_num]
                questions_subject_len = questions_subject_len[:val_num]
                questions_relation_len = questions_relation_len[:val_num]
                questions_object_len = questions_object_len[:val_num]
                video_ids = video_ids[:val_num]
                q_ids = q_ids[:val_num]
                answers = answers[:val_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:val_num]
                    ans_candidates_len = ans_candidates_len[:val_num]
                    ans_bert_candidates = ans_bert_candidates[:val_num]
                    ans_bert_candidates_len = ans_bert_candidates_len[:val_num]
                    ans_bert_attention_mask = ans_bert_attention_mask[:val_num]
                    ans_candidates_token_type = ans_candidates_token_type[:val_num]
        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                questions = questions[:test_num]
                questions_subject = questions_subject[:test_num]
                questions_relation = questions_relation[:test_num]
                questions_object = questions_object[:test_num]

                question_input_bert = question_input_bert[:test_num]        
                question_mask_bert = question_mask_bert[:test_num]
                question_ids_bert = question_ids_bert[:test_num]

                subject_input_bert = subject_input_bert[:test_num]
                subject_mask_bert = subject_mask_bert[:test_num]
                subject_ids_bert = subject_ids_bert[:test_num]

                relation_input_bert = relation_input_bert[:test_num]
                relation_mask_bert = relation_mask_bert[:test_num]
                relation_ids_bert = relation_ids_bert[:test_num]

                object_input_bert = object_input_bert[:test_num]
                object_mask_bert = object_mask_bert[:test_num]
                object_ids_bert = object_ids_bert[:test_num]

                questions_bert_len = questions_bert_len[:test_num]
                questions_subject_bert_len = questions_subject_bert_len[:test_num]   
                questions_relation_bert_len = questions_relation_bert_len[:test_num]
                questions_object_bert_len = questions_object_bert_len[:test_num]  

                questions_len = questions_len[:test_num]
                questions_subject_len = questions_subject_len[:test_num]
                questions_relation_len = questions_relation_len[:test_num]
                questions_object_len = questions_object_len[:test_num]
                video_ids = video_ids[:test_num]
                q_ids = q_ids[:test_num]
                answers = answers[:test_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:test_num]
                    ans_candidates_len = ans_candidates_len[:test_num]
                    ans_bert_candidates = ans_bert_candidates[:test_num]
                    ans_bert_candidates_len = ans_bert_candidates_len[:test_num]
                    ans_bert_attention_mask = ans_bert_attention_mask[:test_num]
                    ans_candidates_token_type = ans_candidates_token_type[:test_num]

        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}

        print('loading motion feature from %s' % (kwargs['motion_feat']))
        with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.motion_feature_h5 = kwargs.pop('motion_feat')
        self.app_dict_h5 = kwargs.pop('appearance_dict')
        self.motion_dict_h5 = kwargs.pop('motion_dict')
        self.dataset = VideoQADataset_oie_bert(answers, ans_candidates, ans_candidates_len, ans_bert_candidates,ans_bert_candidates_len,ans_bert_attention_mask,ans_candidates_token_type,questions, questions_subject, questions_relation, questions_object,\
            question_input_bert,question_mask_bert,question_ids_bert,subject_input_bert,subject_mask_bert,subject_ids_bert,relation_input_bert,relation_mask_bert,relation_ids_bert,object_input_bert,object_mask_bert,object_ids_bert,\
             questions_bert_len,questions_subject_bert_len,questions_relation_bert_len,questions_object_bert_len,questions_len, questions_subject_len, questions_relation_len, questions_object_len, video_ids, q_ids, self.app_feature_h5, app_feat_id_to_index, self.motion_feature_h5, motion_feat_id_to_index, self.app_dict_h5, self.motion_dict_h5)

        self.vocab = vocab
        self.vocab_subject = vocab_subject
        self.vocab_relation = vocab_relation
        self.vocab_object = vocab_object

        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix
        self.glove_matrix_subject = glove_matrix_subject
        self.glove_matrix_relation = glove_matrix_relation
        self.glove_matrix_object = glove_matrix_object

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

class VideoQADataLoader_oie_bert_opendended(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        vocab_subject_json_path = str(kwargs.pop('vocab_subject_json'))
        print('loading vocab_subject from %s' % (vocab_subject_json_path))
        vocab_subject = load_vocab(vocab_subject_json_path)

        vocab_relation_json_path = str(kwargs.pop('vocab_relation_json'))
        print('loading vocab_relation from %s' % (vocab_relation_json_path))
        vocab_relation = load_vocab(vocab_relation_json_path)

        vocab_object_json_path = str(kwargs.pop('vocab_object_json'))
        print('loading vocab_object from %s' % (vocab_object_json_path))
        vocab_object = load_vocab(vocab_object_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')

        question_subject_pt_path = str(kwargs.pop('question_subject_pt'))
        print('loading questions_subject from %s' % (question_subject_pt_path))

        question_relation_pt_path = str(kwargs.pop('question_relation_pt'))
        print('loading questions_relation from %s' % (question_relation_pt_path))

        question_object_pt_path = str(kwargs.pop('question_object_pt'))
        print('loading questions_object from %s' % (question_object_pt_path))

        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            question_input_bert = obj['question_input_bert']
            question_mask_bert = obj['question_mask_bert']
            question_ids_bert = obj['question_ids_bert']
            questions_bert_len = obj['question_bert_len']
            questions_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            # ans_bert_candidates = obj['ans_bert_candidates']
            # ans_bert_candidates_len = obj['ans_candidates_bert_len']
            # ans_bert_attention_mask = obj['ans_bert_attention_mask']
            # ans_candidates_token_type = obj['ans_candidates_token_type']
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']
                # ans_bert_candidates = obj['ans_bert_candidates']
                # ans_bert_candidates_len = obj['ans_candidates_bert_len']
                # ans_bert_attention_mask = obj['ans_bert_attention_mask']
                # ans_candidates_token_type = obj['ans_candidates_token_type']

        with open(question_subject_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_subject = obj['questions']
            subject_input_bert = obj['question_input_bert']
            subject_mask_bert = obj['question_mask_bert']
            subject_ids_bert = obj['question_ids_bert']
            questions_subject_bert_len = obj['question_bert_len']
            questions_subject_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_subject = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            # ans_bert_candidates = obj['ans_bert_candidates']
            # ans_bert_candidates_len = obj['ans_candidates_bert_len']
            # ans_bert_attention_mask = obj['ans_bert_attention_mask']
            # ans_candidates_token_type = obj['ans_candidates_token_type']
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']
                # ans_bert_candidates = obj['ans_bert_candidates']
                # ans_bert_candidates_len = obj['ans_candidates_bert_len']
                # ans_bert_attention_mask = obj['ans_bert_attention_mask']
                # ans_candidates_token_type = obj['ans_candidates_token_type']

        with open(question_relation_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_relation = obj['questions']
            relation_input_bert = obj['question_input_bert']
            relation_mask_bert = obj['question_mask_bert']
            relation_ids_bert = obj['question_ids_bert']
            questions_relation_bert_len = obj['question_bert_len']
            questions_relation_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_relation = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            # ans_bert_candidates = obj['ans_bert_candidates']
            # ans_bert_candidates_len = obj['ans_candidates_bert_len']
            # ans_bert_attention_mask = obj['ans_bert_attention_mask']
            # ans_candidates_token_type = obj['ans_candidates_token_type']
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']
                # ans_bert_candidates = obj['ans_bert_candidates']
                # ans_bert_candidates_len = obj['ans_candidates_bert_len']
                # ans_bert_attention_mask = obj['ans_bert_attention_mask']
                # ans_candidates_token_type = obj['ans_candidates_token_type']

        with open(question_object_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions_object = obj['questions']
            object_input_bert = obj['question_input_bert']
            object_mask_bert = obj['question_mask_bert']
            object_ids_bert = obj['question_ids_bert']
            questions_object_bert_len = obj['question_bert_len']
            questions_object_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix_object = obj['glove']
            ans_candidates = np.zeros(5)
            ans_candidates_len = np.zeros(5)
            # ans_bert_candidates = obj['ans_bert_candidates']
            # ans_bert_candidates_len = obj['ans_candidates_bert_len']
            # ans_bert_attention_mask = obj['ans_bert_attention_mask']
            # ans_candidates_token_type = obj['ans_candidates_token_type']
            if question_type in ['action', 'transition', 'mulchoices']:
                ans_candidates = obj['ans_candidates']
                ans_candidates_len = obj['ans_candidates_len']
                # ans_bert_candidates = obj['ans_bert_candidates']
                # ans_bert_candidates_len = obj['ans_candidates_bert_len']
                # ans_bert_attention_mask = obj['ans_bert_attention_mask']
                # ans_candidates_token_type = obj['ans_candidates_token_type']

        if 'train_num' in kwargs:
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                questions = questions[:trained_num]
                questions_subject = questions_subject[:trained_num]
                questions_relation = questions_relation[:trained_num]
                questions_object = questions_object[:trained_num]

                question_input_bert = question_input_bert[:trained_num]
                question_mask_bert = question_mask_bert[:trained_num]
                question_ids_bert = question_ids_bert[:trained_num]

                subject_input_bert = subject_input_bert[:trained_num]
                subject_mask_bert = subject_mask_bert[:trained_num]
                subject_ids_bert = subject_ids_bert[:trained_num]

                relation_input_bert = relation_input_bert[:trained_num]
                relation_mask_bert = relation_mask_bert[:trained_num]
                relation_ids_bert = relation_ids_bert[:trained_num]

                object_input_bert = object_input_bert[:trained_num]
                object_mask_bert = object_mask_bert[:trained_num]
                object_ids_bert = object_ids_bert[:trained_num]

                questions_bert_len = questions_bert_len[:trained_num]
                questions_subject_bert_len = questions_subject_bert_len[:trained_num]
                questions_relation_bert_len = questions_relation_bert_len[:trained_num]
                questions_object_bert_len = questions_object_bert_len[:trained_num]  

                questions_len = questions_len[:trained_num]
                questions_subject_len = questions_subject_len[:trained_num]
                questions_relation_len = questions_relation_len[:trained_num]
                questions_object_len = questions_object_len[:trained_num]                                                
                video_ids = video_ids[:trained_num]
                q_ids = q_ids[:trained_num]
                answers = answers[:trained_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:trained_num]
                    ans_candidates_len = ans_candidates_len[:trained_num]
                    # ans_bert_candidates = ans_bert_candidates[:trained_num]
                    # ans_bert_candidates_len = ans_bert_candidates_len[:trained_num]
                    # ans_bert_attention_mask = ans_bert_attention_mask[:trained_num]
                    # ans_candidates_token_type = ans_candidates_token_type[:trained_num]
        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                questions = questions[:val_num]
                questions_subject = questions_subject[:val_num]
                questions_relation = questions_relation[:val_num]
                questions_object = questions_object[:val_num]

                question_input_bert = question_input_bert[:val_num]
                question_mask_bert = question_mask_bert[:val_num]
                question_ids_bert = question_ids_bert[:val_num]

                subject_input_bert = subject_input_bert[:val_num]
                subject_mask_bert = subject_mask_bert[:val_num]
                subject_ids_bert = subject_ids_bert[:val_num]

                relation_input_bert = relation_input_bert[:val_num]
                relation_mask_bert = relation_mask_bert[:val_num]
                relation_ids_bert = relation_ids_bert[:val_num]

                object_input_bert = object_input_bert[:val_num]
                object_mask_bert = object_mask_bert[:val_num]
                object_ids_bert = object_ids_bert[:val_num]

                questions_bert_len = questions_bert_len[:val_num]
                questions_subject_bert_len = questions_subject_bert_len[:val_num]
                questions_relation_bert_len = questions_relation_bert_len[:val_num]
                questions_object_bert_len = questions_object_bert_len[:val_num]  

                questions_len = questions_len[:val_num]
                questions_subject_len = questions_subject_len[:val_num]
                questions_relation_len = questions_relation_len[:val_num]
                questions_object_len = questions_object_len[:val_num]
                video_ids = video_ids[:val_num]
                q_ids = q_ids[:val_num]
                answers = answers[:val_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:val_num]
                    ans_candidates_len = ans_candidates_len[:val_num]
                    # ans_bert_candidates = ans_bert_candidates[:val_num]
                    # ans_bert_candidates_len = ans_bert_candidates_len[:val_num]
                    # ans_bert_attention_mask = ans_bert_attention_mask[:val_num]
                    # ans_candidates_token_type = ans_candidates_token_type[:val_num]
        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                questions = questions[:test_num]
                questions_subject = questions_subject[:test_num]
                questions_relation = questions_relation[:test_num]
                questions_object = questions_object[:test_num]

                question_input_bert = question_input_bert[:test_num]        
                question_mask_bert = question_mask_bert[:test_num]
                question_ids_bert = question_ids_bert[:test_num]

                subject_input_bert = subject_input_bert[:test_num]
                subject_mask_bert = subject_mask_bert[:test_num]
                subject_ids_bert = subject_ids_bert[:test_num]

                relation_input_bert = relation_input_bert[:test_num]
                relation_mask_bert = relation_mask_bert[:test_num]
                relation_ids_bert = relation_ids_bert[:test_num]

                object_input_bert = object_input_bert[:test_num]
                object_mask_bert = object_mask_bert[:test_num]
                object_ids_bert = object_ids_bert[:test_num]

                questions_bert_len = questions_bert_len[:test_num]
                questions_subject_bert_len = questions_subject_bert_len[:test_num]   
                questions_relation_bert_len = questions_relation_bert_len[:test_num]
                questions_object_bert_len = questions_object_bert_len[:test_num]  

                questions_len = questions_len[:test_num]
                questions_subject_len = questions_subject_len[:test_num]
                questions_relation_len = questions_relation_len[:test_num]
                questions_object_len = questions_object_len[:test_num]
                video_ids = video_ids[:test_num]
                q_ids = q_ids[:test_num]
                answers = answers[:test_num]
                if question_type in ['action', 'transition']:
                    ans_candidates = ans_candidates[:test_num]
                    ans_candidates_len = ans_candidates_len[:test_num]
                    # ans_bert_candidates = ans_bert_candidates[:test_num]
                    # ans_bert_candidates_len = ans_bert_candidates_len[:test_num]
                    # ans_bert_attention_mask = ans_bert_attention_mask[:test_num]
                    # ans_candidates_token_type = ans_candidates_token_type[:test_num]

        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}

        print('loading motion feature from %s' % (kwargs['motion_feat']))
        with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.motion_feature_h5 = kwargs.pop('motion_feat')
        self.app_dict_h5 = kwargs.pop('appearance_dict')
        self.motion_dict_h5 = kwargs.pop('motion_dict')
        self.dataset = VideoQADataset_oie_bert_openended(answers, ans_candidates, ans_candidates_len,questions, questions_subject, questions_relation, questions_object,\
            question_input_bert,question_mask_bert,question_ids_bert,subject_input_bert,subject_mask_bert,subject_ids_bert,relation_input_bert,relation_mask_bert,relation_ids_bert,object_input_bert,object_mask_bert,object_ids_bert,\
             questions_bert_len,questions_subject_bert_len,questions_relation_bert_len,questions_object_bert_len,questions_len, questions_subject_len, questions_relation_len, questions_object_len, video_ids, q_ids, self.app_feature_h5, app_feat_id_to_index, self.motion_feature_h5, motion_feat_id_to_index, self.app_dict_h5, self.motion_dict_h5)

        self.vocab = vocab
        self.vocab_subject = vocab_subject
        self.vocab_relation = vocab_relation
        self.vocab_object = vocab_object

        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix
        self.glove_matrix_subject = glove_matrix_subject
        self.glove_matrix_relation = glove_matrix_relation
        self.glove_matrix_object = glove_matrix_object

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)