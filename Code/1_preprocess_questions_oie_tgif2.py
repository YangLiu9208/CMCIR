import argparse
import numpy as np
import os
import jsonlines
from preprocess.datautils import tgif_qa
from preprocess.datautils import msrvtt_qa
from preprocess.datautils import msvd_qa
from preprocess.datautils import sutd_qa
import nltk
from openie import StanfordOpenIE
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
properties = {
    'openie.affinity_probability_cap': 2/3,
}

if __name__ == '__main__':
    nltk.download('punkt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tgif-qa', choices=['sutd-qa','tgif-qa', 'msrvtt-qa', 'msvd-qa'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt', default="data/glove/glove.840.300d.pkl", type=str,
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str, default='data/TGIF-QA/{}/tgif-qa_{}_{}_questions_keybert_v4.pt')
    parser.add_argument('--output_subject_pt', type=str, default='data/TGIF-QA/{}/tgif-qa_{}_{}_questions_subject_keybert_v4.pt')
    parser.add_argument('--output_relation_pt', type=str, default='data/TGIF-QA/{}/tgif-qa_{}_{}_questions_relation_keybert_v4.pt')
    parser.add_argument('--output_object_pt', type=str, default='data/TGIF-QA/{}/tgif-qa_{}_{}_questions_object_keybert_v4.pt')
    parser.add_argument('--vocab_json', type=str, default='data/TGIF-QA/{}/tgif-qa_{}_vocab_keybert_v4.json')
    parser.add_argument('--vocab_subject_json', type=str, default='data/TGIF-QA/{}/tgif-qa_{}_vocab_subject_keybert_v4.json')
    parser.add_argument('--vocab_relation_json', type=str, default='data/TGIF-QA/{}/tgif-qa_{}_vocab_relation_keybert_v4.json')
    parser.add_argument('--vocab_object_json', type=str, default='data/TGIF-QA/{}/tgif-qa_{}_vocab_object_keybert_v4.json')
    parser.add_argument('--mode', default='test',choices=['train', 'val', 'test'])
    parser.add_argument('--question_type', choices=['frameqa', 'action', 'transition', 'count', 'none'], default='action')
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset == 'sutd-qa':
        args.annotation_file = 'datasets/SUTD-TrafficQA/annotations/R3_{}.jsonl'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        sutd_qa.process_questions_oie(args)
    elif args.dataset == 'tgif-qa':
        args.annotation_file = 'datasets/TGIF-QA/csv/{}_{}_question.csv'
        #args.output_pt = 'data/TGIF-QA/{}/tgif-qa_{}_{}_questions.pt'
        #args.vocab_json = 'data/TGIF-QA/{}/tgif-qa_{}_vocab.json'
        # check if data folder exists
        if not os.path.exists('data/TGIF-QA/{}'.format(args.question_type)):
            os.makedirs('data/TGIF-QA/{}'.format(args.question_type))

        if args.question_type in ['frameqa', 'count']:
            tgif_qa.process_questions_openended_oie_keybert2(args)
        else:
            tgif_qa.process_questions_mulchoices_oie_keybert2(args)
    elif args.dataset == 'msrvtt-qa':
        args.annotation_file = 'datasets/msrvtt/annotations/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msrvtt_qa.process_questions(args)
    elif args.dataset == 'msvd-qa':
        args.annotation_file = 'datasets/msvd/MSVD-QA/{}_qa.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        msvd_qa.process_questions(args)