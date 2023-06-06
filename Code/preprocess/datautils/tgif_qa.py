import os
import pandas as pd
import json
from preprocess.datautils import utils
import nltk
import torch
import pickle
import numpy as np
import jsonlines
from openie import StanfordOpenIE
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")

def load_video_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    input_paths = []
    annotation = pd.read_csv(args.annotation_file.format(args.question_type), delimiter='\t')
    gif_names = list(annotation['gif_name'])
    keys = list(annotation['key'])
    print("Number of questions: {}".format(len(gif_names)))
    for idx, gif in enumerate(gif_names):
        gif_abs_path = os.path.join(args.video_dir, ''.join([gif, '.gif']))
        input_paths.append((gif_abs_path, keys[idx]))
    input_paths = list(set(input_paths))
    print("Number of unique videos: {}".format(len(input_paths)))

    return input_paths


def openeded_encoding_data(args, vocab, questions, video_names, video_ids, answers, mode='train'):
    ''' Encode question tokens'''
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    question_ids = []
    for idx, question in enumerate(questions):
        question = question.lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)
        video_names_tbw.append(video_names[idx])
        video_ids_tbw.append(video_ids[idx])

        if args.question_type == "frameqa":
            answer = answers[idx]
            if answer in vocab['answer_token_to_idx']:
                answer = vocab['answer_token_to_idx'][answer]
            elif mode in ['train']:
                answer = 0
            elif mode in ['val', 'test']:
                answer = 1
        else:
            answer = max(int(answers[idx]), 1)
        all_answers.append(answer)

    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    glove_matrix = None
    if mode == 'train':
        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'answers': all_answers,
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
        pickle.dump(obj, f)

def openeded_encoding_data_oie(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers, mode='train'):
    ''' Encode question tokens'''
    print('Encoding data')
    questions_encoded_subject = []
    questions_encoded_relation = [] 
    questions_encoded_object = [] 
    questions_len_subject = []
    questions_len_relation = []
    questions_len_object = []
    questions_encoded = []
    questions_len = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    question_ids = []
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    with StanfordOpenIE(properties=properties) as client:
        for idx, question in enumerate(questions):
            question = question.lower()[:-1]
            if client.annotate(question)==[]:
                q=question.split()
                token_subject = ' '.join(q[0:round(len(q)/3)])
                token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                token_object = ' '.join(q[2*round(len(q)/3):len(q)])
            else:
                token_subject = client.annotate(question)[0]['subject']
                token_relation = client.annotate(question)[0]['relation']
                token_object = client.annotate(question)[0]['object']

            subject_tokens = nltk.word_tokenize(token_subject)
            relation_tokens = nltk.word_tokenize(token_relation)
            object_tokens = nltk.word_tokenize(token_object)
            question_subject_encoded = utils.encode(subject_tokens, vocab_subject['question_token_to_idx'], allow_unk=True)
            question_relation_encoded = utils.encode(relation_tokens, vocab_relation['question_token_to_idx'], allow_unk=True)
            question_object_encoded = utils.encode(object_tokens, vocab_object['question_token_to_idx'], allow_unk=True)

            questions_encoded_subject.append(question_subject_encoded)
            questions_len_subject.append(len(question_subject_encoded))
            questions_encoded_relation.append(question_relation_encoded)
            questions_len_relation.append(len(question_relation_encoded))
            questions_encoded_object.append(question_object_encoded)
            questions_len_object.append(len(question_object_encoded))

            question_tokens = nltk.word_tokenize(question)
            question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            question_ids.append(idx)
            video_names_tbw.append(video_names[idx])
            video_ids_tbw.append(video_ids[idx])

            if args.question_type == "frameqa":
                answer = answers[idx]
                if answer in vocab['answer_token_to_idx']:
                    answer = vocab['answer_token_to_idx'][answer]
                elif mode in ['train']:
                    answer = 0
                elif mode in ['val', 'test']:
                    answer = 1
            else:
                answer = max(int(answers[idx]), 1)
            all_answers.append(answer)

        # Pad encoded questions
        max_question_subject_length = max(len(x) for x in questions_encoded_subject)
        for qe in questions_encoded_subject:
            while len(qe) < max_question_subject_length:
                qe.append(vocab_subject['question_token_to_idx']['<NULL>'])

        max_question_relation_length = max(len(x) for x in questions_encoded_relation)
        for qe in questions_encoded_relation:
            while len(qe) < max_question_relation_length:
                qe.append(vocab_relation['question_token_to_idx']['<NULL>'])

        max_question_object_length = max(len(x) for x in questions_encoded_object)
        for qe in questions_encoded_object:
            while len(qe) < max_question_object_length:
                qe.append(vocab_object['question_token_to_idx']['<NULL>'])

        max_question_length = max(len(x) for x in questions_encoded)
        for qe in questions_encoded:
            while len(qe) < max_question_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        questions_encoded_subject = np.asarray(questions_encoded_subject, dtype=np.int32)
        questions_len_subject = np.asarray(questions_len_subject, dtype=np.int32)
        print(questions_encoded_subject.shape)

        questions_encoded_relation = np.asarray(questions_encoded_relation, dtype=np.int32)
        questions_len_relation = np.asarray(questions_len_relation, dtype=np.int32)
        print(questions_encoded_relation.shape)

        questions_encoded_object = np.asarray(questions_encoded_object, dtype=np.int32)
        questions_len_object = np.asarray(questions_len_object, dtype=np.int32)
        print(questions_encoded_object.shape)

        questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
        questions_len = np.asarray(questions_len, dtype=np.int32)
        print(questions_encoded.shape)

        glove_matrix = None
        glove_matrix_subject = None
        glove_matrix_relation = None
        glove_matrix_object = None
        if mode == 'train':
            token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
            token_itow_subject = {i: w for w, i in vocab_subject['question_token_to_idx'].items()}
            token_itow_relation = {i: w for w, i in vocab_relation['question_token_to_idx'].items()}
            token_itow_object = {i: w for w, i in vocab_object['question_token_to_idx'].items()}
            print("Load glove from %s" % args.glove_pt)
            glove = pickle.load(open(args.glove_pt, 'rb'))
            dim_word = glove['the'].shape[0]
            glove_matrix = []
            glove_matrix_subject = []
            glove_matrix_relation = []
            glove_matrix_object = []
            for i in range(len(token_itow)):
                vector = glove.get(token_itow[i], np.zeros((dim_word,)))
                glove_matrix.append(vector)
            glove_matrix = np.asarray(glove_matrix, dtype=np.float32)

            for i in range(len(token_itow_subject)):
                vector = glove.get(token_itow_subject[i], np.zeros((dim_word,)))
                glove_matrix_subject.append(vector)
            glove_matrix_subject = np.asarray(glove_matrix_subject, dtype=np.float32)

            for i in range(len(token_itow_relation)):
                vector = glove.get(token_itow_relation[i], np.zeros((dim_word,)))
                glove_matrix_relation.append(vector)
            glove_matrix_relation = np.asarray(glove_matrix_relation, dtype=np.float32)

            for i in range(len(token_itow_object)):
                vector = glove.get(token_itow_object[i], np.zeros((dim_word,)))
                glove_matrix_object.append(vector)
            glove_matrix_object = np.asarray(glove_matrix_object, dtype=np.float32)

            print(glove_matrix.shape)

        print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded,
            'questions_len': questions_len,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'answers': all_answers,
            'glove': glove_matrix,
        }
        with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_subject_pt.format(args.question_type, args.question_type, mode))
        obj = {
                'questions': questions_encoded_subject,
                'questions_len': questions_len_subject,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_subject,
            }
        with open(args.output_subject_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_relation_pt.format(args.question_type, args.question_type, mode))
        obj = {
                'questions': questions_encoded_relation,
                'questions_len': questions_len_relation,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_relation,
            }
        with open(args.output_relation_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_object_pt.format(args.question_type, args.question_type, mode))
        obj = {
                'questions': questions_encoded_object,
                'questions_len': questions_len_object,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_object,
            }
        with open(args.output_object_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

def openeded_encoding_data_oie_keybert(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers, mode='train'):
    ''' Encode question tokens'''
    print('Encoding data')
    questions_encoded_subject = []
    questions_encoded_relation = [] 
    questions_encoded_object = [] 
    questions_len_subject = []
    questions_len_relation = []
    questions_len_object = []
    questions_encoded = []
    questions_len = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    question_ids = []
    kw_model = KeyBERT()  
    # roberta = TransformerDocumentEmbeddings('roberta-base')
    # kw_model = KeyBERT(model=roberta)
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    with StanfordOpenIE(properties=properties) as client:
        for idx, question in enumerate(questions):
            question = question.lower()[:-1]
            keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(2, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
            print(keywords)
            if keywords==[]:
                q=question.split()
                token_subject = ' '.join(q[0:round(len(q)/3)])
                token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                token_object = ' '.join(q[2*round(len(q)/3):len(q)])
            else:
                token_subject = ''.join(keywords[0][0])
                token_relation =''.join(keywords[1][0])
                token_object = ''.join(keywords[2][0])

            subject_tokens = nltk.word_tokenize(token_subject)
            relation_tokens = nltk.word_tokenize(token_relation)
            object_tokens = nltk.word_tokenize(token_object)
            question_subject_encoded = utils.encode(subject_tokens, vocab_subject['question_token_to_idx'], allow_unk=True)
            question_relation_encoded = utils.encode(relation_tokens, vocab_relation['question_token_to_idx'], allow_unk=True)
            question_object_encoded = utils.encode(object_tokens, vocab_object['question_token_to_idx'], allow_unk=True)

            questions_encoded_subject.append(question_subject_encoded)
            questions_len_subject.append(len(question_subject_encoded))
            questions_encoded_relation.append(question_relation_encoded)
            questions_len_relation.append(len(question_relation_encoded))
            questions_encoded_object.append(question_object_encoded)
            questions_len_object.append(len(question_object_encoded))

            question_tokens = nltk.word_tokenize(question)
            question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            question_ids.append(idx)
            video_names_tbw.append(video_names[idx])
            video_ids_tbw.append(video_ids[idx])

            if args.question_type == "frameqa":
                answer = answers[idx]
                if answer in vocab['answer_token_to_idx']:
                    answer = vocab['answer_token_to_idx'][answer]
                elif mode in ['train']:
                    answer = 0
                elif mode in ['val', 'test']:
                    answer = 1
            else:
                answer = max(int(answers[idx]), 1)
            all_answers.append(answer)

        # Pad encoded questions
        max_question_subject_length = max(len(x) for x in questions_encoded_subject)
        for qe in questions_encoded_subject:
            while len(qe) < max_question_subject_length:
                qe.append(vocab_subject['question_token_to_idx']['<NULL>'])

        max_question_relation_length = max(len(x) for x in questions_encoded_relation)
        for qe in questions_encoded_relation:
            while len(qe) < max_question_relation_length:
                qe.append(vocab_relation['question_token_to_idx']['<NULL>'])

        max_question_object_length = max(len(x) for x in questions_encoded_object)
        for qe in questions_encoded_object:
            while len(qe) < max_question_object_length:
                qe.append(vocab_object['question_token_to_idx']['<NULL>'])

        max_question_length = max(len(x) for x in questions_encoded)
        for qe in questions_encoded:
            while len(qe) < max_question_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        questions_encoded_subject = np.asarray(questions_encoded_subject, dtype=np.int32)
        questions_len_subject = np.asarray(questions_len_subject, dtype=np.int32)
        print(questions_encoded_subject.shape)

        questions_encoded_relation = np.asarray(questions_encoded_relation, dtype=np.int32)
        questions_len_relation = np.asarray(questions_len_relation, dtype=np.int32)
        print(questions_encoded_relation.shape)

        questions_encoded_object = np.asarray(questions_encoded_object, dtype=np.int32)
        questions_len_object = np.asarray(questions_len_object, dtype=np.int32)
        print(questions_encoded_object.shape)

        questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
        questions_len = np.asarray(questions_len, dtype=np.int32)
        print(questions_encoded.shape)

        glove_matrix = None
        glove_matrix_subject = None
        glove_matrix_relation = None
        glove_matrix_object = None
        if mode == 'train':
            token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
            token_itow_subject = {i: w for w, i in vocab_subject['question_token_to_idx'].items()}
            token_itow_relation = {i: w for w, i in vocab_relation['question_token_to_idx'].items()}
            token_itow_object = {i: w for w, i in vocab_object['question_token_to_idx'].items()}
            print("Load glove from %s" % args.glove_pt)
            glove = pickle.load(open(args.glove_pt, 'rb'))
            dim_word = glove['the'].shape[0]
            glove_matrix = []
            glove_matrix_subject = []
            glove_matrix_relation = []
            glove_matrix_object = []
            for i in range(len(token_itow)):
                vector = glove.get(token_itow[i], np.zeros((dim_word,)))
                glove_matrix.append(vector)
            glove_matrix = np.asarray(glove_matrix, dtype=np.float32)

            for i in range(len(token_itow_subject)):
                vector = glove.get(token_itow_subject[i], np.zeros((dim_word,)))
                glove_matrix_subject.append(vector)
            glove_matrix_subject = np.asarray(glove_matrix_subject, dtype=np.float32)

            for i in range(len(token_itow_relation)):
                vector = glove.get(token_itow_relation[i], np.zeros((dim_word,)))
                glove_matrix_relation.append(vector)
            glove_matrix_relation = np.asarray(glove_matrix_relation, dtype=np.float32)

            for i in range(len(token_itow_object)):
                vector = glove.get(token_itow_object[i], np.zeros((dim_word,)))
                glove_matrix_object.append(vector)
            glove_matrix_object = np.asarray(glove_matrix_object, dtype=np.float32)

            print(glove_matrix.shape)

        print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded,
            'questions_len': questions_len,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'answers': all_answers,
            'glove': glove_matrix,
        }
        with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_subject_pt.format(args.question_type, args.question_type, mode))
        obj = {
                'questions': questions_encoded_subject,
                'questions_len': questions_len_subject,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_subject,
            }
        with open(args.output_subject_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_relation_pt.format(args.question_type, args.question_type, mode))
        obj = {
                'questions': questions_encoded_relation,
                'questions_len': questions_len_relation,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_relation,
            }
        with open(args.output_relation_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_object_pt.format(args.question_type, args.question_type, mode))
        obj = {
                'questions': questions_encoded_object,
                'questions_len': questions_len_object,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_object,
            }
        with open(args.output_object_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

def openeded_encoding_data_oie_bert3(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers, mode='train'):
    ''' Encode question tokens'''
    print('Encoding data')
    questions_encoded_subject = []
    questions_encoded_relation = [] 
    questions_encoded_object = [] 
    subject_input_batch_bert= []
    subject_attention_mask_batch_bert= []
    subject_token_type_ids_batch_bert= []
    relation_input_batch_bert= []
    relation_attention_mask_batch_bert= []
    relation_token_type_ids_batch_bert= []
    object_input_batch_bert= []
    object_attention_mask_batch_bert= []
    object_token_type_ids_batch_bert= []
    question_input_batch_bert= []
    question_attention_mask_batch_bert= []
    question_token_type_ids_batch_bert= []
    questions_len_subject = []
    questions_len_relation = []
    questions_len_object = []
    subject_bert_len = []
    relation_bert_len = []
    object_bert_len = []
    questions_encoded = []
    questions_len = []
    question_bert_len = []
    question_ids = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    question_ids = []
    kw_model = KeyBERT()  
    # roberta = TransformerDocumentEmbeddings('roberta-base')
    # kw_model = KeyBERT(model=roberta)
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    with StanfordOpenIE(properties=properties) as client:
        for idx, question in enumerate(questions):
            question = question.lower()[:-1]
            #keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(2, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
            #print(keywords)
            if client.annotate(question)==[]:
                q=question.split()
                token_subject = ' '.join(q[0:round(len(q)/3)])
                token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                token_object = ' '.join(q[2*round(len(q)/3):len(q)])
            else:
                token_subject = client.annotate(question)[0]['subject']
                token_relation = client.annotate(question)[0]['relation']
                token_object = client.annotate(question)[0]['object']

            subject_tokens = nltk.word_tokenize(token_subject)
            relation_tokens = nltk.word_tokenize(token_relation)
            object_tokens = nltk.word_tokenize(token_object)
            question_subject_encoded = utils.encode(subject_tokens, vocab_subject['question_token_to_idx'], allow_unk=True)
            question_relation_encoded = utils.encode(relation_tokens, vocab_relation['question_token_to_idx'], allow_unk=True)
            question_object_encoded = utils.encode(object_tokens, vocab_object['question_token_to_idx'], allow_unk=True)

            subject_tokens_dict = tokenizer([token_subject], padding=True)
            subject_input_batch = subject_tokens_dict["input_ids"]
            subject_attention_mask_batch = subject_tokens_dict["attention_mask"]
            subject_token_type_ids_batch = subject_tokens_dict["token_type_ids"]

            relation_tokens_dict = tokenizer([token_relation], padding=True)
            relation_input_batch = relation_tokens_dict["input_ids"]
            relation_attention_mask_batch = relation_tokens_dict["attention_mask"]
            relation_token_type_ids_batch = relation_tokens_dict["token_type_ids"]
            object_tokens_dict = tokenizer([token_object], padding=True)
            object_input_batch = object_tokens_dict["input_ids"]
            object_attention_mask_batch = object_tokens_dict["attention_mask"]
            object_token_type_ids_batch = object_tokens_dict["token_type_ids"]

            questions_encoded_subject.append(question_subject_encoded)
            questions_len_subject.append(len(question_subject_encoded))
            questions_encoded_relation.append(question_relation_encoded)
            questions_len_relation.append(len(question_relation_encoded))
            questions_encoded_object.append(question_object_encoded)
            questions_len_object.append(len(question_object_encoded))

            subject_input_batch_bert.append(subject_input_batch)
            subject_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(subject_input_batch))).squeeze().size())
            subject_attention_mask_batch_bert.append(subject_attention_mask_batch)
            subject_token_type_ids_batch_bert.append(subject_token_type_ids_batch)
            relation_input_batch_bert.append(relation_input_batch)
            relation_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(relation_input_batch))).squeeze().size())
            relation_attention_mask_batch_bert.append(relation_attention_mask_batch)
            relation_token_type_ids_batch_bert.append(relation_token_type_ids_batch)
            object_input_batch_bert.append(object_input_batch)
            object_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(object_input_batch))).squeeze().size())
            object_attention_mask_batch_bert.append(object_attention_mask_batch)
            object_token_type_ids_batch_bert.append(object_token_type_ids_batch)

            question_tokens = nltk.word_tokenize(question)
            question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
            question_tokens_dict = tokenizer([question], padding=True)
            question_input_batch = question_tokens_dict["input_ids"]
            question_attention_mask_batch = question_tokens_dict["attention_mask"]
            question_token_type_ids_batch = question_tokens_dict["token_type_ids"]

            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            question_input_batch_bert.append(question_input_batch)
            question_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(question_input_batch))).squeeze().size())
            question_attention_mask_batch_bert.append(question_attention_mask_batch)
            question_token_type_ids_batch_bert.append(question_token_type_ids_batch)
            question_ids.append(idx)
            video_names_tbw.append(video_names[idx])
            video_ids_tbw.append(video_ids[idx])

            if args.question_type == "frameqa":
                answer = answers[idx]
                if answer in vocab['answer_token_to_idx']:
                    answer = vocab['answer_token_to_idx'][answer]
                elif mode in ['train']:
                    answer = 0
                elif mode in ['val', 'test']:
                    answer = 1
            else:
                answer = max(int(answers[idx]), 1)
            all_answers.append(answer)

        # Pad encoded questions
        max_question_subject_length = max(len(x) for x in questions_encoded_subject)
        for qe in questions_encoded_subject:
            while len(qe) < max_question_subject_length:
                qe.append(vocab_subject['question_token_to_idx']['<NULL>'])

        max_question_relation_length = max(len(x) for x in questions_encoded_relation)
        for qe in questions_encoded_relation:
            while len(qe) < max_question_relation_length:
                qe.append(vocab_relation['question_token_to_idx']['<NULL>'])

        max_question_object_length = max(len(x) for x in questions_encoded_object)
        for qe in questions_encoded_object:
            while len(qe) < max_question_object_length:
                qe.append(vocab_object['question_token_to_idx']['<NULL>'])

        max_question_length = max(len(x) for x in questions_encoded)
        for qe in questions_encoded:
            while len(qe) < max_question_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_subject_input_batch_bert_length = max(len(x) for x in subject_input_batch_bert)
        for qe in subject_input_batch_bert:
            while len(qe) < max_subject_input_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_subject_attention_mask_batch_bert_length = max(len(x) for x in subject_attention_mask_batch_bert)
        for qe in subject_attention_mask_batch_bert:
            while len(qe) < max_subject_attention_mask_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_subject_token_type_ids_batch_bert_length = max(len(x) for x in subject_token_type_ids_batch_bert)
        for qe in subject_token_type_ids_batch_bert:
            while len(qe) < max_subject_token_type_ids_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_relation_input_batch_bert_length = max(len(x) for x in relation_input_batch_bert)
        for qe in relation_input_batch_bert:
            while len(qe) < max_relation_input_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_relation_attention_mask_batch_bert_length = max(len(x) for x in relation_attention_mask_batch_bert)
        for qe in relation_attention_mask_batch_bert:
            while len(qe) < max_relation_attention_mask_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_relation_token_type_ids_batch_bert_length = max(len(x) for x in relation_token_type_ids_batch_bert)
        for qe in relation_token_type_ids_batch_bert:
            while len(qe) < max_relation_token_type_ids_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_object_input_batch_bert_length = max(len(x) for x in object_input_batch_bert)
        for qe in object_input_batch_bert:
            while len(qe) < max_object_input_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_object_attention_mask_batch_bert_length = max(len(x) for x in object_attention_mask_batch_bert)
        for qe in object_attention_mask_batch_bert:
            while len(qe) < max_object_attention_mask_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_object_token_type_ids_batch_bert_length = max(len(x) for x in object_token_type_ids_batch_bert)
        for qe in object_token_type_ids_batch_bert:
            while len(qe) < max_object_token_type_ids_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])    

        questions_encoded_subject = np.asarray(questions_encoded_subject, dtype=np.int32)
        questions_len_subject = np.asarray(questions_len_subject, dtype=np.int32)
        print(questions_encoded_subject.shape)

        questions_encoded_relation = np.asarray(questions_encoded_relation, dtype=np.int32)
        questions_len_relation = np.asarray(questions_len_relation, dtype=np.int32)
        print(questions_encoded_relation.shape)

        questions_encoded_object = np.asarray(questions_encoded_object, dtype=np.int32)
        questions_len_object = np.asarray(questions_len_object, dtype=np.int32)
        print(questions_encoded_object.shape)

        questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
        questions_len = np.asarray(questions_len, dtype=np.int32)
        print(questions_encoded.shape)

        glove_matrix = None
        glove_matrix_subject = None
        glove_matrix_relation = None
        glove_matrix_object = None
        if mode == 'train':
            token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
            token_itow_subject = {i: w for w, i in vocab_subject['question_token_to_idx'].items()}
            token_itow_relation = {i: w for w, i in vocab_relation['question_token_to_idx'].items()}
            token_itow_object = {i: w for w, i in vocab_object['question_token_to_idx'].items()}
            print("Load glove from %s" % args.glove_pt)
            glove = pickle.load(open(args.glove_pt, 'rb'))
            dim_word = glove['the'].shape[0]
            glove_matrix = []
            glove_matrix_subject = []
            glove_matrix_relation = []
            glove_matrix_object = []
            for i in range(len(token_itow)):
                vector = glove.get(token_itow[i], np.zeros((dim_word,)))
                glove_matrix.append(vector)
            glove_matrix = np.asarray(glove_matrix, dtype=np.float32)

            for i in range(len(token_itow_subject)):
                vector = glove.get(token_itow_subject[i], np.zeros((dim_word,)))
                glove_matrix_subject.append(vector)
            glove_matrix_subject = np.asarray(glove_matrix_subject, dtype=np.float32)

            for i in range(len(token_itow_relation)):
                vector = glove.get(token_itow_relation[i], np.zeros((dim_word,)))
                glove_matrix_relation.append(vector)
            glove_matrix_relation = np.asarray(glove_matrix_relation, dtype=np.float32)

            for i in range(len(token_itow_object)):
                vector = glove.get(token_itow_object[i], np.zeros((dim_word,)))
                glove_matrix_object.append(vector)
            glove_matrix_object = np.asarray(glove_matrix_object, dtype=np.float32)

            print(glove_matrix.shape)

        print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded,
            'question_input_bert': question_input_batch_bert,   
            'question_mask_bert': question_attention_mask_batch_bert,
            'question_ids_bert': question_token_type_ids_batch_bert,        
            'question_bert_len': question_bert_len, 
            'questions_len': questions_len,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'answers': all_answers,
            'glove': glove_matrix,
        }
        with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_subject_pt.format(args.question_type, args.question_type, mode))
        obj = {
                'questions': questions_encoded_subject,
                'question_input_bert': question_input_batch_bert,   
                'question_mask_bert': question_attention_mask_batch_bert,
                'question_ids_bert': question_token_type_ids_batch_bert,        
                'question_bert_len': question_bert_len, 
                'questions_len': questions_len_subject,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_subject,
            }
        with open(args.output_subject_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_relation_pt.format(args.question_type, args.question_type, mode))
        obj = {
                'questions': questions_encoded_relation,
                'question_input_bert': question_input_batch_bert,   
                'question_mask_bert': question_attention_mask_batch_bert,
                'question_ids_bert': question_token_type_ids_batch_bert,        
                'question_bert_len': question_bert_len, 
                'questions_len': questions_len_relation,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_relation,
            }
        with open(args.output_relation_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_object_pt.format(args.question_type, args.question_type, mode))
        obj = {
                'questions': questions_encoded_object,
                'question_input_bert': question_input_batch_bert,   
                'question_mask_bert': question_attention_mask_batch_bert,
                'question_ids_bert': question_token_type_ids_batch_bert,        
                'question_bert_len': question_bert_len, 
                'questions_len': questions_len_object,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_object,
            }
        with open(args.output_object_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

def openeded_encoding_data_oie_bert2(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers, mode='train'):
    ''' Encode question tokens'''
    print('Encoding data')
    questions_encoded_subject = []
    questions_encoded_relation = [] 
    questions_encoded_object = [] 
    subject_input_batch_bert= []
    subject_attention_mask_batch_bert= []
    subject_token_type_ids_batch_bert= []
    relation_input_batch_bert= []
    relation_attention_mask_batch_bert= []
    relation_token_type_ids_batch_bert= []
    object_input_batch_bert= []
    object_attention_mask_batch_bert= []
    object_token_type_ids_batch_bert= []
    question_input_batch_bert= []
    question_attention_mask_batch_bert= []
    question_token_type_ids_batch_bert= []
    questions_len_subject = []
    questions_len_relation = []
    questions_len_object = []
    subject_bert_len = []
    relation_bert_len = []
    object_bert_len = []
    questions_encoded = []
    questions_len = []
    question_bert_len = []
    question_ids = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    question_ids = []
    kw_model = KeyBERT()  
    # roberta = TransformerDocumentEmbeddings('roberta-base')
    # kw_model = KeyBERT(model=roberta)
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    with StanfordOpenIE(properties=properties) as client:
        for idx, question in enumerate(questions):
            question = question.lower()[:-1]
            keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(2, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
            print(keywords)
            if keywords==[]:
                q=question.split()
                token_subject = ' '.join(q[0:round(len(q)/3)])
                token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                token_object = ' '.join(q[2*round(len(q)/3):len(q)])
            else:
                token_subject = ''.join(keywords[0][0])
                token_relation =''.join(keywords[1][0])
                token_object = ''.join(keywords[2][0])

            subject_tokens = nltk.word_tokenize(token_subject)
            relation_tokens = nltk.word_tokenize(token_relation)
            object_tokens = nltk.word_tokenize(token_object)
            question_subject_encoded = utils.encode(subject_tokens, vocab_subject['question_token_to_idx'], allow_unk=True)
            question_relation_encoded = utils.encode(relation_tokens, vocab_relation['question_token_to_idx'], allow_unk=True)
            question_object_encoded = utils.encode(object_tokens, vocab_object['question_token_to_idx'], allow_unk=True)

            subject_tokens_dict = tokenizer([token_subject], padding=True)
            subject_input_batch = subject_tokens_dict["input_ids"]
            subject_attention_mask_batch = subject_tokens_dict["attention_mask"]
            subject_token_type_ids_batch = subject_tokens_dict["token_type_ids"]

            relation_tokens_dict = tokenizer([token_relation], padding=True)
            relation_input_batch = relation_tokens_dict["input_ids"]
            relation_attention_mask_batch = relation_tokens_dict["attention_mask"]
            relation_token_type_ids_batch = relation_tokens_dict["token_type_ids"]
            object_tokens_dict = tokenizer([token_object], padding=True)
            object_input_batch = object_tokens_dict["input_ids"]
            object_attention_mask_batch = object_tokens_dict["attention_mask"]
            object_token_type_ids_batch = object_tokens_dict["token_type_ids"]

            questions_encoded_subject.append(question_subject_encoded)
            questions_len_subject.append(len(question_subject_encoded))
            questions_encoded_relation.append(question_relation_encoded)
            questions_len_relation.append(len(question_relation_encoded))
            questions_encoded_object.append(question_object_encoded)
            questions_len_object.append(len(question_object_encoded))

            subject_input_batch_bert.append(subject_input_batch)
            subject_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(subject_input_batch))).squeeze().size())
            subject_attention_mask_batch_bert.append(subject_attention_mask_batch)
            subject_token_type_ids_batch_bert.append(subject_token_type_ids_batch)
            relation_input_batch_bert.append(relation_input_batch)
            relation_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(relation_input_batch))).squeeze().size())
            relation_attention_mask_batch_bert.append(relation_attention_mask_batch)
            relation_token_type_ids_batch_bert.append(relation_token_type_ids_batch)
            object_input_batch_bert.append(object_input_batch)
            object_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(object_input_batch))).squeeze().size())
            object_attention_mask_batch_bert.append(object_attention_mask_batch)
            object_token_type_ids_batch_bert.append(object_token_type_ids_batch)

            question_tokens = nltk.word_tokenize(question)
            question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
            question_tokens_dict = tokenizer([question], padding=True)
            question_input_batch = question_tokens_dict["input_ids"]
            question_attention_mask_batch = question_tokens_dict["attention_mask"]
            question_token_type_ids_batch = question_tokens_dict["token_type_ids"]

            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            question_input_batch_bert.append(question_input_batch)
            question_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(question_input_batch))).squeeze().size())
            question_attention_mask_batch_bert.append(question_attention_mask_batch)
            question_token_type_ids_batch_bert.append(question_token_type_ids_batch)
            question_ids.append(idx)
            video_names_tbw.append(video_names[idx])
            video_ids_tbw.append(video_ids[idx])

            if args.question_type == "frameqa":
                answer = answers[idx]
                if answer in vocab['answer_token_to_idx']:
                    answer = vocab['answer_token_to_idx'][answer]
                elif mode in ['train']:
                    answer = 0
                elif mode in ['val', 'test']:
                    answer = 1
            else:
                answer = max(int(answers[idx]), 1)
            all_answers.append(answer)

        # Pad encoded questions
        max_question_subject_length = max(len(x) for x in questions_encoded_subject)
        for qe in questions_encoded_subject:
            while len(qe) < max_question_subject_length:
                qe.append(vocab_subject['question_token_to_idx']['<NULL>'])

        max_question_relation_length = max(len(x) for x in questions_encoded_relation)
        for qe in questions_encoded_relation:
            while len(qe) < max_question_relation_length:
                qe.append(vocab_relation['question_token_to_idx']['<NULL>'])

        max_question_object_length = max(len(x) for x in questions_encoded_object)
        for qe in questions_encoded_object:
            while len(qe) < max_question_object_length:
                qe.append(vocab_object['question_token_to_idx']['<NULL>'])

        max_question_length = max(len(x) for x in questions_encoded)
        for qe in questions_encoded:
            while len(qe) < max_question_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_subject_input_batch_bert_length = max(len(x) for x in subject_input_batch_bert)
        for qe in subject_input_batch_bert:
            while len(qe) < max_subject_input_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_subject_attention_mask_batch_bert_length = max(len(x) for x in subject_attention_mask_batch_bert)
        for qe in subject_attention_mask_batch_bert:
            while len(qe) < max_subject_attention_mask_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_subject_token_type_ids_batch_bert_length = max(len(x) for x in subject_token_type_ids_batch_bert)
        for qe in subject_token_type_ids_batch_bert:
            while len(qe) < max_subject_token_type_ids_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_relation_input_batch_bert_length = max(len(x) for x in relation_input_batch_bert)
        for qe in relation_input_batch_bert:
            while len(qe) < max_relation_input_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_relation_attention_mask_batch_bert_length = max(len(x) for x in relation_attention_mask_batch_bert)
        for qe in relation_attention_mask_batch_bert:
            while len(qe) < max_relation_attention_mask_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_relation_token_type_ids_batch_bert_length = max(len(x) for x in relation_token_type_ids_batch_bert)
        for qe in relation_token_type_ids_batch_bert:
            while len(qe) < max_relation_token_type_ids_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_object_input_batch_bert_length = max(len(x) for x in object_input_batch_bert)
        for qe in object_input_batch_bert:
            while len(qe) < max_object_input_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_object_attention_mask_batch_bert_length = max(len(x) for x in object_attention_mask_batch_bert)
        for qe in object_attention_mask_batch_bert:
            while len(qe) < max_object_attention_mask_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_object_token_type_ids_batch_bert_length = max(len(x) for x in object_token_type_ids_batch_bert)
        for qe in object_token_type_ids_batch_bert:
            while len(qe) < max_object_token_type_ids_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])    

        questions_encoded_subject = np.asarray(questions_encoded_subject, dtype=np.int32)
        questions_len_subject = np.asarray(questions_len_subject, dtype=np.int32)
        print(questions_encoded_subject.shape)

        questions_encoded_relation = np.asarray(questions_encoded_relation, dtype=np.int32)
        questions_len_relation = np.asarray(questions_len_relation, dtype=np.int32)
        print(questions_encoded_relation.shape)

        questions_encoded_object = np.asarray(questions_encoded_object, dtype=np.int32)
        questions_len_object = np.asarray(questions_len_object, dtype=np.int32)
        print(questions_encoded_object.shape)

        questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
        questions_len = np.asarray(questions_len, dtype=np.int32)
        print(questions_encoded.shape)

        glove_matrix = None
        glove_matrix_subject = None
        glove_matrix_relation = None
        glove_matrix_object = None
        if mode == 'train':
            token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
            token_itow_subject = {i: w for w, i in vocab_subject['question_token_to_idx'].items()}
            token_itow_relation = {i: w for w, i in vocab_relation['question_token_to_idx'].items()}
            token_itow_object = {i: w for w, i in vocab_object['question_token_to_idx'].items()}
            print("Load glove from %s" % args.glove_pt)
            glove = pickle.load(open(args.glove_pt, 'rb'))
            dim_word = glove['the'].shape[0]
            glove_matrix = []
            glove_matrix_subject = []
            glove_matrix_relation = []
            glove_matrix_object = []
            for i in range(len(token_itow)):
                vector = glove.get(token_itow[i], np.zeros((dim_word,)))
                glove_matrix.append(vector)
            glove_matrix = np.asarray(glove_matrix, dtype=np.float32)

            for i in range(len(token_itow_subject)):
                vector = glove.get(token_itow_subject[i], np.zeros((dim_word,)))
                glove_matrix_subject.append(vector)
            glove_matrix_subject = np.asarray(glove_matrix_subject, dtype=np.float32)

            for i in range(len(token_itow_relation)):
                vector = glove.get(token_itow_relation[i], np.zeros((dim_word,)))
                glove_matrix_relation.append(vector)
            glove_matrix_relation = np.asarray(glove_matrix_relation, dtype=np.float32)

            for i in range(len(token_itow_object)):
                vector = glove.get(token_itow_object[i], np.zeros((dim_word,)))
                glove_matrix_object.append(vector)
            glove_matrix_object = np.asarray(glove_matrix_object, dtype=np.float32)

            print(glove_matrix.shape)

        print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded,
            'question_input_bert': question_input_batch_bert,   
            'question_mask_bert': question_attention_mask_batch_bert,
            'question_ids_bert': question_token_type_ids_batch_bert,        
            'question_bert_len': question_bert_len, 
            'questions_len': questions_len,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'answers': all_answers,
            'glove': glove_matrix,
        }
        with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_subject_pt.format(args.question_type, args.question_type, mode))
        obj = {
                'questions': questions_encoded_subject,
                'question_input_bert': question_input_batch_bert,   
                'question_mask_bert': question_attention_mask_batch_bert,
                'question_ids_bert': question_token_type_ids_batch_bert,        
                'question_bert_len': question_bert_len, 
                'questions_len': questions_len_subject,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_subject,
            }
        with open(args.output_subject_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_relation_pt.format(args.question_type, args.question_type, mode))
        obj = {
                'questions': questions_encoded_relation,
                'question_input_bert': question_input_batch_bert,   
                'question_mask_bert': question_attention_mask_batch_bert,
                'question_ids_bert': question_token_type_ids_batch_bert,        
                'question_bert_len': question_bert_len, 
                'questions_len': questions_len_relation,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_relation,
            }
        with open(args.output_relation_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_object_pt.format(args.question_type, args.question_type, mode))
        obj = {
                'questions': questions_encoded_object,
                'question_input_bert': question_input_batch_bert,   
                'question_mask_bert': question_attention_mask_batch_bert,
                'question_ids_bert': question_token_type_ids_batch_bert,        
                'question_bert_len': question_bert_len, 
                'questions_len': questions_len_object,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_object,
            }
        with open(args.output_object_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

def multichoice_encoding_data(args, vocab, questions, video_names, video_ids, answers, ans_candidates, mode='train'):
    # Encode all questions
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    question_ids = []
    all_answer_cands_encoded = []
    all_answer_cands_len = []
    video_ids_tbw = []
    video_names_tbw = []
    correct_answers = []
    for idx, question in enumerate(questions):
        question = question.lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)
        video_names_tbw.append(video_names[idx])
        video_ids_tbw.append(video_ids[idx])
        # grounthtruth
        answer = int(answers[idx])
        correct_answers.append(answer)
        # answer candidates
        candidates = ans_candidates[idx]
        candidates_encoded = []
        candidates_len = []
        for ans in candidates:
            ans = ans.lower()
            ans_tokens = nltk.word_tokenize(ans)
            cand_encoded = utils.encode(ans_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
            candidates_encoded.append(cand_encoded)
            candidates_len.append(len(cand_encoded))
        all_answer_cands_encoded.append(candidates_encoded)
        all_answer_cands_len.append(candidates_len)

    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_answer_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    # Pad encoded answer candidates
    max_answer_cand_length = max(max(len(x) for x in candidate) for candidate in all_answer_cands_encoded)
    for ans_cands in all_answer_cands_encoded:
        for ans in ans_cands:
            while len(ans) < max_answer_cand_length:
                ans.append(vocab['question_answer_token_to_idx']['<NULL>'])
    all_answer_cands_encoded = np.asarray(all_answer_cands_encoded, dtype=np.int32)
    all_answer_cands_len = np.asarray(all_answer_cands_len, dtype=np.int32)
    print(all_answer_cands_encoded.shape)

    glove_matrix = None
    if mode in ['train']:
        token_itow = {i: w for w, i in vocab['question_answer_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'ans_candidates': all_answer_cands_encoded,
        'ans_candidates_len': all_answer_cands_len,
        'answers': correct_answers,
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
        pickle.dump(obj, f)

def multichoice_encoding_data_oie(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers, ans_candidates, mode='train'):
    # Encode all questions
    print('Encoding data')
    questions_encoded_subject = []
    questions_encoded_relation = [] 
    questions_encoded_object = [] 
    questions_len_subject = []
    questions_len_relation = []
    questions_len_object = []
    questions_encoded = []
    questions_len = []
    question_ids = []
    all_answer_cands_encoded = []
    all_answer_cands_len = []
    video_ids_tbw = []
    video_names_tbw = []
    correct_answers = []
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    with StanfordOpenIE(properties=properties) as client:
        for idx, question in enumerate(questions):
            question = question.lower()[:-1]
            if client.annotate(question)==[]:
                q=question.split()
                token_subject = ' '.join(q[0:round(len(q)/3)])
                token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                token_object = ' '.join(q[2*round(len(q)/3):len(q)])
            else:
                token_subject = client.annotate(question)[0]['subject']
                token_relation = client.annotate(question)[0]['relation']
                token_object = client.annotate(question)[0]['object']

            subject_tokens = nltk.word_tokenize(token_subject)
            relation_tokens = nltk.word_tokenize(token_relation)
            object_tokens = nltk.word_tokenize(token_object)
            question_subject_encoded = utils.encode(subject_tokens, vocab_subject['question_answer_token_to_idx'], allow_unk=True)
            question_relation_encoded = utils.encode(relation_tokens, vocab_relation['question_answer_token_to_idx'], allow_unk=True)
            question_object_encoded = utils.encode(object_tokens, vocab_object['question_answer_token_to_idx'], allow_unk=True)
            questions_encoded_subject.append(question_subject_encoded)
            questions_len_subject.append(len(question_subject_encoded))
            questions_encoded_relation.append(question_relation_encoded)
            questions_len_relation.append(len(question_relation_encoded))
            questions_encoded_object.append(question_object_encoded)
            questions_len_object.append(len(question_object_encoded))

            question_tokens = nltk.word_tokenize(question)
            question_encoded = utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            question_ids.append(idx)
            video_names_tbw.append(video_names[idx])
            video_ids_tbw.append(video_ids[idx])
            # grounthtruth
            answer = int(answers[idx])
            correct_answers.append(answer)
            # answer candidates
            candidates = ans_candidates[idx]
            candidates_encoded = []
            candidates_len = []
            for ans in candidates:
                ans = ans.lower()
                ans_tokens = nltk.word_tokenize(ans)
                cand_encoded = utils.encode(ans_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
                candidates_encoded.append(cand_encoded)
                candidates_len.append(len(cand_encoded))
            all_answer_cands_encoded.append(candidates_encoded)
            all_answer_cands_len.append(candidates_len)

        # Pad encoded questions
        max_question_subject_length = max(len(x) for x in questions_encoded_subject)
        for qe in questions_encoded_subject:
            while len(qe) < max_question_subject_length:
                qe.append(vocab_subject['question_answer_token_to_idx']['<NULL>'])

        max_question_relation_length = max(len(x) for x in questions_encoded_relation)
        for qe in questions_encoded_relation:
            while len(qe) < max_question_relation_length:
                qe.append(vocab_relation['question_answer_token_to_idx']['<NULL>'])

        max_question_object_length = max(len(x) for x in questions_encoded_object)
        for qe in questions_encoded_object:
            while len(qe) < max_question_object_length:
                qe.append(vocab_object['question_answer_token_to_idx']['<NULL>'])

        max_question_length = max(len(x) for x in questions_encoded)
        for qe in questions_encoded:
            while len(qe) < max_question_length:
                qe.append(vocab['question_answer_token_to_idx']['<NULL>'])

        questions_encoded_subject = np.asarray(questions_encoded_subject, dtype=np.int32)
        questions_len_subject = np.asarray(questions_len_subject, dtype=np.int32)
        print(questions_encoded_subject.shape)

        questions_encoded_relation = np.asarray(questions_encoded_relation, dtype=np.int32)
        questions_len_relation = np.asarray(questions_len_relation, dtype=np.int32)
        print(questions_encoded_relation.shape)

        questions_encoded_object = np.asarray(questions_encoded_object, dtype=np.int32)
        questions_len_object = np.asarray(questions_len_object, dtype=np.int32)
        print(questions_encoded_object.shape)

        questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
        questions_len = np.asarray(questions_len, dtype=np.int32)
        print(questions_encoded.shape)

        # Pad encoded answer candidates
        max_answer_cand_length = max(max(len(x) for x in candidate) for candidate in all_answer_cands_encoded)
        for ans_cands in all_answer_cands_encoded:
            for ans in ans_cands:
                while len(ans) < max_answer_cand_length:
                    ans.append(vocab['question_answer_token_to_idx']['<NULL>'])
        all_answer_cands_encoded = np.asarray(all_answer_cands_encoded, dtype=np.int32)
        all_answer_cands_len = np.asarray(all_answer_cands_len, dtype=np.int32)
        print(all_answer_cands_encoded.shape)

        glove_matrix = None
        glove_matrix_subject = None
        glove_matrix_relation = None
        glove_matrix_object = None
        if mode in ['train']:
            token_itow = {i: w for w, i in vocab['question_answer_token_to_idx'].items()}
            token_itow_subject = {i: w for w, i in vocab_subject['question_answer_token_to_idx'].items()}
            token_itow_relation = {i: w for w, i in vocab_relation['question_answer_token_to_idx'].items()}
            token_itow_object = {i: w for w, i in vocab_object['question_answer_token_to_idx'].items()}
            print("Load glove from %s" % args.glove_pt)
            glove = pickle.load(open(args.glove_pt, 'rb'))
            dim_word = glove['the'].shape[0]
            glove_matrix = []
            glove_matrix_subject = []
            glove_matrix_relation = []
            glove_matrix_object = []
            for i in range(len(token_itow)):
                vector = glove.get(token_itow[i], np.zeros((dim_word,)))
                glove_matrix.append(vector)
            glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
            print(glove_matrix.shape)

            for i in range(len(token_itow_subject)):
                vector = glove.get(token_itow_subject[i], np.zeros((dim_word,)))
                glove_matrix_subject.append(vector)
            glove_matrix_subject = np.asarray(glove_matrix_subject, dtype=np.float32)

            for i in range(len(token_itow_relation)):
                vector = glove.get(token_itow_relation[i], np.zeros((dim_word,)))
                glove_matrix_relation.append(vector)
            glove_matrix_relation = np.asarray(glove_matrix_relation, dtype=np.float32)

            for i in range(len(token_itow_object)):
                vector = glove.get(token_itow_object[i], np.zeros((dim_word,)))
                glove_matrix_object.append(vector)
            glove_matrix_object = np.asarray(glove_matrix_object, dtype=np.float32)
            print(glove_matrix_object.shape)

        print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded,
            'questions_len': questions_len,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'answers': correct_answers,
            'glove': glove_matrix,
        }
        with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing ', args.output_subject_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded_subject,
            'questions_len': questions_len_subject,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'answers': correct_answers,
            'glove': glove_matrix_subject,
        }
        with open(args.output_subject_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing ', args.output_relation_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded_relation,
            'questions_len': questions_len_relation,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'answers': correct_answers,
            'glove': glove_matrix_relation,
        }
        with open(args.output_relation_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing ', args.output_object_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded_object,
            'questions_len': questions_len_object,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'answers': correct_answers,
            'glove': glove_matrix_object,
        }
        with open(args.output_object_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)


def multichoice_encoding_data_oie_bert2(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers, ans_candidates, mode='train'):
    # Encode all questions
    print('Encoding data')
    questions_encoded_subject = []
    questions_encoded_relation = [] 
    questions_encoded_object = [] 
    subject_input_batch_bert= []
    subject_attention_mask_batch_bert= []
    subject_token_type_ids_batch_bert= []
    relation_input_batch_bert= []
    relation_attention_mask_batch_bert= []
    relation_token_type_ids_batch_bert= []
    object_input_batch_bert= []
    object_attention_mask_batch_bert= []
    object_token_type_ids_batch_bert= []
    question_input_batch_bert= []
    question_attention_mask_batch_bert= []
    question_token_type_ids_batch_bert= []
    questions_len_subject = []
    questions_len_relation = []
    questions_len_object = []
    subject_bert_len = []
    relation_bert_len = []
    object_bert_len = []
    questions_encoded = []
    questions_len = []
    question_bert_len = []
    question_ids = []
    all_answer_cands_encoded = []
    all_answer_cands_len = []
    all_candidateds_bert_encoded = []
    all_candidates_bert_len = []
    all_candidates_attention_mask_batch_bert = []
    all_candidates_token_type_ids_batch_bert = []
    video_ids_tbw = []
    video_names_tbw = []
    correct_answers = []
    kw_model = KeyBERT() 
    # roberta = TransformerDocumentEmbeddings('roberta-base')
    # kw_model = KeyBERT(model=roberta) 
    properties = {
        'openie.affinity_probability_cap':2/3,
                        }
    with StanfordOpenIE(properties=properties) as client:
        for idx, question in enumerate(questions):
            question = question.lower()[:-1]
            #keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(2, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
            #print(keywords)
            # if keywords==[]:
            if client.annotate(question)==[]:
                q=question.split()
                token_subject = ' '.join(q[0:round(len(q)/3)])
                token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                token_object = ' '.join(q[2*round(len(q)/3):len(q)])
            else:
                # token_subject = ''.join(keywords[0][0])
                # token_relation =''.join(keywords[1][0])
                # token_object = ''.join(keywords[2][0])
                token_subject = client.annotate(question)[0]['subject']
                token_relation = client.annotate(question)[0]['relation']
                token_object = client.annotate(question)[0]['object']

            subject_tokens = nltk.word_tokenize(token_subject)
            relation_tokens = nltk.word_tokenize(token_relation)
            object_tokens = nltk.word_tokenize(token_object)
            question_subject_encoded = utils.encode(subject_tokens, vocab_subject['question_answer_token_to_idx'], allow_unk=True)
            question_relation_encoded = utils.encode(relation_tokens, vocab_relation['question_answer_token_to_idx'], allow_unk=True)
            question_object_encoded = utils.encode(object_tokens, vocab_object['question_answer_token_to_idx'], allow_unk=True)
            
            subject_tokens_dict = tokenizer([token_subject], padding=True)
            subject_input_batch = subject_tokens_dict["input_ids"]
            subject_attention_mask_batch = subject_tokens_dict["attention_mask"]
            subject_token_type_ids_batch = subject_tokens_dict["token_type_ids"]

            relation_tokens_dict = tokenizer([token_relation], padding=True)
            relation_input_batch = relation_tokens_dict["input_ids"]
            relation_attention_mask_batch = relation_tokens_dict["attention_mask"]
            relation_token_type_ids_batch = relation_tokens_dict["token_type_ids"]
            object_tokens_dict = tokenizer([token_object], padding=True)
            object_input_batch = object_tokens_dict["input_ids"]
            object_attention_mask_batch = object_tokens_dict["attention_mask"]
            object_token_type_ids_batch = object_tokens_dict["token_type_ids"]

            questions_encoded_subject.append(question_subject_encoded)
            questions_len_subject.append(len(question_subject_encoded))
            questions_encoded_relation.append(question_relation_encoded)
            questions_len_relation.append(len(question_relation_encoded))
            questions_encoded_object.append(question_object_encoded)
            questions_len_object.append(len(question_object_encoded))

            subject_input_batch_bert.append(subject_input_batch)
            subject_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(subject_input_batch))).squeeze().size())
            subject_attention_mask_batch_bert.append(subject_attention_mask_batch)
            subject_token_type_ids_batch_bert.append(subject_token_type_ids_batch)
            relation_input_batch_bert.append(relation_input_batch)
            relation_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(relation_input_batch))).squeeze().size())
            relation_attention_mask_batch_bert.append(relation_attention_mask_batch)
            relation_token_type_ids_batch_bert.append(relation_token_type_ids_batch)
            object_input_batch_bert.append(object_input_batch)
            object_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(object_input_batch))).squeeze().size())
            object_attention_mask_batch_bert.append(object_attention_mask_batch)
            object_token_type_ids_batch_bert.append(object_token_type_ids_batch)

            question_tokens = nltk.word_tokenize(question)
            question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
            question_tokens_dict = tokenizer([question], padding=True)
            question_input_batch = question_tokens_dict["input_ids"]
            question_attention_mask_batch = question_tokens_dict["attention_mask"]
            question_token_type_ids_batch = question_tokens_dict["token_type_ids"]

            question_tokens = nltk.word_tokenize(question)
            question_encoded = utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
            question_input_batch_bert.append(question_input_batch)
            question_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(question_input_batch))).squeeze().size())
            question_attention_mask_batch_bert.append(question_attention_mask_batch)
            question_token_type_ids_batch_bert.append(question_token_type_ids_batch)            
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            question_ids.append(idx)
            video_names_tbw.append(video_names[idx])
            video_ids_tbw.append(video_ids[idx])
            # grounthtruth
            answer = int(answers[idx])
            correct_answers.append(answer)
            # answer candidates
            candidates = ans_candidates[idx]
            candidates_encoded = []
            candidates_len = []
            candidateds_bert_encoded = []
            candidates_bert_len = []
            candidates_attention_mask_batch_bert = []
            candidates_token_type_ids_batch_bert = []
            for ans in candidates:
                ans = ans.lower()
                ans_tokens = nltk.word_tokenize(ans)
                cand_encoded = utils.encode(ans_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
                candidates_encoded.append(cand_encoded)
                candidates_len.append(len(cand_encoded))
                answer_tokens_dict = tokenizer([ans], padding=True)
                answer_input_batch = answer_tokens_dict["input_ids"]
                answer_attention_mask_batch = answer_tokens_dict["attention_mask"]
                answer_token_type_ids_batch = answer_tokens_dict["token_type_ids"]
                candidateds_bert_encoded.append(answer_input_batch)
                candidates_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(answer_input_batch))).squeeze().size())
                candidates_attention_mask_batch_bert.append(answer_attention_mask_batch)
                candidates_token_type_ids_batch_bert.append(answer_token_type_ids_batch)
            all_answer_cands_encoded.append(candidates_encoded)
            all_answer_cands_len.append(candidates_len)
            all_candidateds_bert_encoded.append(candidateds_bert_encoded)
            all_candidates_bert_len.append(candidates_bert_len)
            all_candidates_attention_mask_batch_bert.append(candidates_attention_mask_batch_bert)
            all_candidates_token_type_ids_batch_bert.append(candidates_token_type_ids_batch_bert)

        # Pad encoded questions
        max_question_subject_length = max(len(x) for x in questions_encoded_subject)
        for qe in questions_encoded_subject:
            while len(qe) < max_question_subject_length:
                qe.append(vocab_subject['question_answer_token_to_idx']['<NULL>'])

        max_question_relation_length = max(len(x) for x in questions_encoded_relation)
        for qe in questions_encoded_relation:
            while len(qe) < max_question_relation_length:
                qe.append(vocab_relation['question_answer_token_to_idx']['<NULL>'])

        max_question_object_length = max(len(x) for x in questions_encoded_object)
        for qe in questions_encoded_object:
            while len(qe) < max_question_object_length:
                qe.append(vocab_object['question_answer_token_to_idx']['<NULL>'])

        max_question_length = max(len(x) for x in questions_encoded)
        for qe in questions_encoded:
            while len(qe) < max_question_length:
                qe.append(vocab['question_answer_token_to_idx']['<NULL>'])

        max_subject_input_batch_bert_length = max(len(x) for x in subject_input_batch_bert)
        for qe in subject_input_batch_bert:
            while len(qe) < max_subject_input_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_subject_attention_mask_batch_bert_length = max(len(x) for x in subject_attention_mask_batch_bert)
        for qe in subject_attention_mask_batch_bert:
            while len(qe) < max_subject_attention_mask_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_subject_token_type_ids_batch_bert_length = max(len(x) for x in subject_token_type_ids_batch_bert)
        for qe in subject_token_type_ids_batch_bert:
            while len(qe) < max_subject_token_type_ids_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_relation_input_batch_bert_length = max(len(x) for x in relation_input_batch_bert)
        for qe in relation_input_batch_bert:
            while len(qe) < max_relation_input_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_relation_attention_mask_batch_bert_length = max(len(x) for x in relation_attention_mask_batch_bert)
        for qe in relation_attention_mask_batch_bert:
            while len(qe) < max_relation_attention_mask_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_relation_token_type_ids_batch_bert_length = max(len(x) for x in relation_token_type_ids_batch_bert)
        for qe in relation_token_type_ids_batch_bert:
            while len(qe) < max_relation_token_type_ids_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_object_input_batch_bert_length = max(len(x) for x in object_input_batch_bert)
        for qe in object_input_batch_bert:
            while len(qe) < max_object_input_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_object_attention_mask_batch_bert_length = max(len(x) for x in object_attention_mask_batch_bert)
        for qe in object_attention_mask_batch_bert:
            while len(qe) < max_object_attention_mask_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        max_object_token_type_ids_batch_bert_length = max(len(x) for x in object_token_type_ids_batch_bert)
        for qe in object_token_type_ids_batch_bert:
            while len(qe) < max_object_token_type_ids_batch_bert_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])    

        questions_encoded_subject = np.asarray(questions_encoded_subject, dtype=np.int32)
        questions_len_subject = np.asarray(questions_len_subject, dtype=np.int32)
        print(questions_encoded_subject.shape)

        questions_encoded_relation = np.asarray(questions_encoded_relation, dtype=np.int32)
        questions_len_relation = np.asarray(questions_len_relation, dtype=np.int32)
        print(questions_encoded_relation.shape)

        questions_encoded_object = np.asarray(questions_encoded_object, dtype=np.int32)
        questions_len_object = np.asarray(questions_len_object, dtype=np.int32)
        print(questions_encoded_object.shape)

        questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
        questions_len = np.asarray(questions_len, dtype=np.int32)
        print(questions_encoded.shape)

        # Pad encoded answer candidates
        max_answer_cand_length = max(max(len(x) for x in candidate) for candidate in all_answer_cands_encoded)
        for ans_cands in all_answer_cands_encoded:
            for ans in ans_cands:
                while len(ans) < max_answer_cand_length:
                    ans.append(vocab['question_answer_token_to_idx']['<NULL>'])
        max_answer_cand_length = max(max(len(x[0]) for x in candidate) for candidate in all_candidateds_bert_encoded)
        for ans_cands in all_candidateds_bert_encoded:
            for ans in ans_cands:
                for ans_s in ans:
                    while len(ans_s) < max_answer_cand_length:
                        ans_s.append(vocab['question_answer_token_to_idx']['<NULL>'])
        max_answer_cand_length = max(max(len(x[0]) for x in candidate) for candidate in all_candidates_attention_mask_batch_bert)
        for ans_cands in all_candidates_attention_mask_batch_bert:
            for ans in ans_cands:
                for ans_s in ans:
                    while len(ans_s) < max_answer_cand_length:
                        ans_s.append(vocab['question_answer_token_to_idx']['<NULL>'])
        max_answer_cand_length = max(max(len(x[0]) for x in candidate) for candidate in all_candidates_token_type_ids_batch_bert)
        for ans_cands in all_candidates_token_type_ids_batch_bert:
            for ans in ans_cands:
                for ans_s in ans:
                    while len(ans_s) < max_answer_cand_length:
                        ans_s.append(vocab['question_answer_token_to_idx']['<NULL>'])

        all_answer_cands_encoded = np.asarray(all_answer_cands_encoded, dtype=np.int32)
        all_answer_cands_len = np.asarray(all_answer_cands_len, dtype=np.int32)
        all_candidateds_bert_encoded= all_candidateds_bert_encoded
        all_candidates_attention_mask_batch_bert=all_candidates_attention_mask_batch_bert
        all_candidates_token_type_ids_batch_bert=all_candidates_token_type_ids_batch_bert
        print(all_answer_cands_encoded.shape)

        glove_matrix = None
        glove_matrix_subject = None
        glove_matrix_relation = None
        glove_matrix_object = None
        if mode in ['train']:
            token_itow = {i: w for w, i in vocab['question_answer_token_to_idx'].items()}
            token_itow_subject = {i: w for w, i in vocab_subject['question_answer_token_to_idx'].items()}
            token_itow_relation = {i: w for w, i in vocab_relation['question_answer_token_to_idx'].items()}
            token_itow_object = {i: w for w, i in vocab_object['question_answer_token_to_idx'].items()}
            print("Load glove from %s" % args.glove_pt)
            glove = pickle.load(open(args.glove_pt, 'rb'))
            dim_word = glove['the'].shape[0]
            glove_matrix = []
            glove_matrix_subject = []
            glove_matrix_relation = []
            glove_matrix_object = []
            for i in range(len(token_itow)):
                vector = glove.get(token_itow[i], np.zeros((dim_word,)))
                glove_matrix.append(vector)
            glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
            print(glove_matrix.shape)

            for i in range(len(token_itow_subject)):
                vector = glove.get(token_itow_subject[i], np.zeros((dim_word,)))
                glove_matrix_subject.append(vector)
            glove_matrix_subject = np.asarray(glove_matrix_subject, dtype=np.float32)

            for i in range(len(token_itow_relation)):
                vector = glove.get(token_itow_relation[i], np.zeros((dim_word,)))
                glove_matrix_relation.append(vector)
            glove_matrix_relation = np.asarray(glove_matrix_relation, dtype=np.float32)

            for i in range(len(token_itow_object)):
                vector = glove.get(token_itow_object[i], np.zeros((dim_word,)))
                glove_matrix_object.append(vector)
            glove_matrix_object = np.asarray(glove_matrix_object, dtype=np.float32)
            print(glove_matrix_object.shape)

        print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded,
            'question_input_bert': question_input_batch_bert,   
            'question_mask_bert': question_attention_mask_batch_bert,
            'question_ids_bert': question_token_type_ids_batch_bert,        
            'question_bert_len': question_bert_len, 
            'ans_bert_candidates': all_candidateds_bert_encoded,
            'ans_candidates_bert_len': all_candidates_bert_len,
            'ans_bert_attention_mask': all_candidates_attention_mask_batch_bert,                
            'ans_candidates_token_type': all_candidates_token_type_ids_batch_bert,
            'questions_len': questions_len,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'answers': correct_answers,
            'glove': glove_matrix,
        }
        with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing ', args.output_subject_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded_subject,
            'question_input_bert': question_input_batch_bert,   
            'question_mask_bert': question_attention_mask_batch_bert,
            'question_ids_bert': question_token_type_ids_batch_bert,        
            'question_bert_len': question_bert_len,  
            'ans_bert_candidates': all_candidateds_bert_encoded,
            'ans_candidates_bert_len': all_candidates_bert_len,
            'ans_bert_attention_mask': all_candidates_attention_mask_batch_bert,                
            'ans_candidates_token_type': all_candidates_token_type_ids_batch_bert,
            'questions_len': questions_len_subject,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'answers': correct_answers,
            'glove': glove_matrix_subject,
        }
        with open(args.output_subject_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing ', args.output_relation_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded_relation,
            'question_input_bert': question_input_batch_bert,   
            'question_mask_bert': question_attention_mask_batch_bert,
            'question_ids_bert': question_token_type_ids_batch_bert,        
            'question_bert_len': question_bert_len,  
            'ans_bert_candidates': all_candidateds_bert_encoded,
            'ans_candidates_bert_len': all_candidates_bert_len,
            'ans_bert_attention_mask': all_candidates_attention_mask_batch_bert,                
            'ans_candidates_token_type': all_candidates_token_type_ids_batch_bert,
            'questions_len': questions_len_relation,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'answers': correct_answers,
            'glove': glove_matrix_relation,
        }
        with open(args.output_relation_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing ', args.output_object_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded_object,
            'question_input_bert': question_input_batch_bert,   
            'question_mask_bert': question_attention_mask_batch_bert,
            'question_ids_bert': question_token_type_ids_batch_bert,        
            'question_bert_len': question_bert_len,  
            'ans_bert_candidates': all_candidateds_bert_encoded,
            'ans_candidates_bert_len': all_candidates_bert_len,
            'ans_bert_attention_mask': all_candidates_attention_mask_batch_bert,                
            'ans_candidates_token_type': all_candidates_token_type_ids_batch_bert,
            'questions_len': questions_len_object,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'answers': correct_answers,
            'glove': glove_matrix_object,
        }
        with open(args.output_object_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

def multichoice_encoding_data_oie_keybert(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers, ans_candidates, mode='train'):
    # Encode all questions
    print('Encoding data')
    questions_encoded_subject = []
    questions_encoded_relation = [] 
    questions_encoded_object = [] 
    questions_len_subject = []
    questions_len_relation = []
    questions_len_object = []
    questions_encoded = []
    questions_len = []
    question_ids = []
    all_answer_cands_encoded = []
    all_answer_cands_len = []
    video_ids_tbw = []
    video_names_tbw = []
    correct_answers = []
    kw_model = KeyBERT() 
    # roberta = TransformerDocumentEmbeddings('roberta-base')
    # kw_model = KeyBERT(model=roberta) 
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    with StanfordOpenIE(properties=properties) as client:
        for idx, question in enumerate(questions):
            question = question.lower()[:-1]
            keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(2, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
            print(keywords)
            if keywords==[]:
                q=question.split()
                token_subject = ' '.join(q[0:round(len(q)/3)])
                token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                token_object = ' '.join(q[2*round(len(q)/3):len(q)])
            else:
                token_subject = ''.join(keywords[0][0])
                token_relation =''.join(keywords[1][0])
                token_object = ''.join(keywords[2][0])

            subject_tokens = nltk.word_tokenize(token_subject)
            relation_tokens = nltk.word_tokenize(token_relation)
            object_tokens = nltk.word_tokenize(token_object)
            question_subject_encoded = utils.encode(subject_tokens, vocab_subject['question_answer_token_to_idx'], allow_unk=True)
            question_relation_encoded = utils.encode(relation_tokens, vocab_relation['question_answer_token_to_idx'], allow_unk=True)
            question_object_encoded = utils.encode(object_tokens, vocab_object['question_answer_token_to_idx'], allow_unk=True)
            questions_encoded_subject.append(question_subject_encoded)
            questions_len_subject.append(len(question_subject_encoded))
            questions_encoded_relation.append(question_relation_encoded)
            questions_len_relation.append(len(question_relation_encoded))
            questions_encoded_object.append(question_object_encoded)
            questions_len_object.append(len(question_object_encoded))

            question_tokens = nltk.word_tokenize(question)
            question_encoded = utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            question_ids.append(idx)
            video_names_tbw.append(video_names[idx])
            video_ids_tbw.append(video_ids[idx])
            # grounthtruth
            answer = int(answers[idx])
            correct_answers.append(answer)
            # answer candidates
            candidates = ans_candidates[idx]
            candidates_encoded = []
            candidates_len = []
            for ans in candidates:
                ans = ans.lower()
                ans_tokens = nltk.word_tokenize(ans)
                cand_encoded = utils.encode(ans_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
                candidates_encoded.append(cand_encoded)
                candidates_len.append(len(cand_encoded))
            all_answer_cands_encoded.append(candidates_encoded)
            all_answer_cands_len.append(candidates_len)

        # Pad encoded questions
        max_question_subject_length = max(len(x) for x in questions_encoded_subject)
        for qe in questions_encoded_subject:
            while len(qe) < max_question_subject_length:
                qe.append(vocab_subject['question_answer_token_to_idx']['<NULL>'])

        max_question_relation_length = max(len(x) for x in questions_encoded_relation)
        for qe in questions_encoded_relation:
            while len(qe) < max_question_relation_length:
                qe.append(vocab_relation['question_answer_token_to_idx']['<NULL>'])

        max_question_object_length = max(len(x) for x in questions_encoded_object)
        for qe in questions_encoded_object:
            while len(qe) < max_question_object_length:
                qe.append(vocab_object['question_answer_token_to_idx']['<NULL>'])

        max_question_length = max(len(x) for x in questions_encoded)
        for qe in questions_encoded:
            while len(qe) < max_question_length:
                qe.append(vocab['question_answer_token_to_idx']['<NULL>'])

        questions_encoded_subject = np.asarray(questions_encoded_subject, dtype=np.int32)
        questions_len_subject = np.asarray(questions_len_subject, dtype=np.int32)
        print(questions_encoded_subject.shape)

        questions_encoded_relation = np.asarray(questions_encoded_relation, dtype=np.int32)
        questions_len_relation = np.asarray(questions_len_relation, dtype=np.int32)
        print(questions_encoded_relation.shape)

        questions_encoded_object = np.asarray(questions_encoded_object, dtype=np.int32)
        questions_len_object = np.asarray(questions_len_object, dtype=np.int32)
        print(questions_encoded_object.shape)

        questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
        questions_len = np.asarray(questions_len, dtype=np.int32)
        print(questions_encoded.shape)

        # Pad encoded answer candidates
        max_answer_cand_length = max(max(len(x) for x in candidate) for candidate in all_answer_cands_encoded)
        for ans_cands in all_answer_cands_encoded:
            for ans in ans_cands:
                while len(ans) < max_answer_cand_length:
                    ans.append(vocab['question_answer_token_to_idx']['<NULL>'])
        all_answer_cands_encoded = np.asarray(all_answer_cands_encoded, dtype=np.int32)
        all_answer_cands_len = np.asarray(all_answer_cands_len, dtype=np.int32)
        print(all_answer_cands_encoded.shape)

        glove_matrix = None
        glove_matrix_subject = None
        glove_matrix_relation = None
        glove_matrix_object = None
        if mode in ['train']:
            token_itow = {i: w for w, i in vocab['question_answer_token_to_idx'].items()}
            token_itow_subject = {i: w for w, i in vocab_subject['question_answer_token_to_idx'].items()}
            token_itow_relation = {i: w for w, i in vocab_relation['question_answer_token_to_idx'].items()}
            token_itow_object = {i: w for w, i in vocab_object['question_answer_token_to_idx'].items()}
            print("Load glove from %s" % args.glove_pt)
            glove = pickle.load(open(args.glove_pt, 'rb'))
            dim_word = glove['the'].shape[0]
            glove_matrix = []
            glove_matrix_subject = []
            glove_matrix_relation = []
            glove_matrix_object = []
            for i in range(len(token_itow)):
                vector = glove.get(token_itow[i], np.zeros((dim_word,)))
                glove_matrix.append(vector)
            glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
            print(glove_matrix.shape)

            for i in range(len(token_itow_subject)):
                vector = glove.get(token_itow_subject[i], np.zeros((dim_word,)))
                glove_matrix_subject.append(vector)
            glove_matrix_subject = np.asarray(glove_matrix_subject, dtype=np.float32)

            for i in range(len(token_itow_relation)):
                vector = glove.get(token_itow_relation[i], np.zeros((dim_word,)))
                glove_matrix_relation.append(vector)
            glove_matrix_relation = np.asarray(glove_matrix_relation, dtype=np.float32)

            for i in range(len(token_itow_object)):
                vector = glove.get(token_itow_object[i], np.zeros((dim_word,)))
                glove_matrix_object.append(vector)
            glove_matrix_object = np.asarray(glove_matrix_object, dtype=np.float32)
            print(glove_matrix_object.shape)

        print('Writing ', args.output_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded,
            'questions_len': questions_len,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'answers': correct_answers,
            'glove': glove_matrix,
        }
        with open(args.output_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing ', args.output_subject_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded_subject,
            'questions_len': questions_len_subject,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'answers': correct_answers,
            'glove': glove_matrix_subject,
        }
        with open(args.output_subject_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing ', args.output_relation_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded_relation,
            'questions_len': questions_len_relation,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'answers': correct_answers,
            'glove': glove_matrix_relation,
        }
        with open(args.output_relation_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing ', args.output_object_pt.format(args.question_type, args.question_type, mode))
        obj = {
            'questions': questions_encoded_object,
            'questions_len': questions_len_object,
            'question_id': question_ids,
            'video_ids': np.asarray(video_ids_tbw),
            'video_names': np.array(video_names_tbw),
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'answers': correct_answers,
            'glove': glove_matrix_object,
        }
        with open(args.output_object_pt.format(args.question_type, args.question_type, mode), 'wb') as f:
            pickle.dump(obj, f)

def process_questions_openended(args):
    print('Loading data')
    if args.mode in ["train"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])

    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        print('Building vocab')
        answer_cnt = {}

        if args.question_type == "frameqa":
            for i, answer in enumerate(answers):
                answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

            answer_token_to_idx = {'<UNK>': 0}
            for token in answer_cnt:
                answer_token_to_idx[token] = len(answer_token_to_idx)
            print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))
        elif args.question_type == 'count':
            answer_token_to_idx = {'<UNK>': 0}

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for i, q in enumerate(questions):
            question = q.lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
        print('Get question_token_to_idx')
        print(len(question_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
        }

        print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
        with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
            json.dump(vocab, f, indent=4)

        # split 10% of questions for evaluation
        split = int(0.9 * len(questions))
        train_questions = questions[:split]
        train_answers = answers[:split]
        train_video_names = video_names[:split]
        train_video_ids = video_ids[:split]

        val_questions = questions[split:]
        val_answers = answers[split:]
        val_video_names = video_names[split:]
        val_video_ids = video_ids[split:]

        openeded_encoding_data(args, vocab, train_questions, train_video_names, train_video_ids, train_answers, mode='train')
        openeded_encoding_data(args, vocab, val_questions, val_video_names, val_video_ids, val_answers, mode='val')
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
            vocab = json.load(f)
        openeded_encoding_data(args, vocab, questions, video_names, video_ids, answers, mode='test')


def process_questions_openended_oie_keybert2(args):
    print('Loading data')
    if args.mode in ["train"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    kw_model = KeyBERT()  
    # roberta = TransformerDocumentEmbeddings('roberta-base')
    # kw_model = KeyBERT(model=roberta)
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    with StanfordOpenIE(properties=properties) as client:
        if args.mode in ['train']:
            print('Building vocab')
            answer_cnt = {}

            if args.question_type == "frameqa":
                for i, answer in enumerate(answers):
                    answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

                answer_token_to_idx = {'<UNK>': 0}
                for token in answer_cnt:
                    answer_token_to_idx[token] = len(answer_token_to_idx)
                print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))
            elif args.question_type == 'count':
                answer_token_to_idx = {'<UNK>': 0}

            question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_subject_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_relation_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_object_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            for i, q in enumerate(questions):
                question = q.lower()[:-1]
                keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(2, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
                print(keywords)
                if keywords==[]:
                    q=question.split()
                    token_subject = ' '.join(q[0:round(len(q)/3)])
                    token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                    token_object = ' '.join(q[2*round(len(q)/3):len(q)])
                else:
                    token_subject = ''.join(keywords[0][0])
                    token_relation =''.join(keywords[1][0])
                    token_object = ''.join(keywords[2][0])
                for token in nltk.word_tokenize(token_subject):
                    if token not in question_oie_subject_token_to_idx:
                        question_oie_subject_token_to_idx[token] = len(question_oie_subject_token_to_idx)
                for token in nltk.word_tokenize(token_relation):
                    if token not in question_oie_relation_token_to_idx:
                        question_oie_relation_token_to_idx[token] = len(question_oie_relation_token_to_idx)
                for token in nltk.word_tokenize(token_object):
                    if token not in question_oie_object_token_to_idx:
                        question_oie_object_token_to_idx[token] = len(question_oie_object_token_to_idx)
                for token in nltk.word_tokenize(question):
                    if token not in question_token_to_idx:
                        question_token_to_idx[token] = len(question_token_to_idx)
            print('Get question_token_to_idx')
            print(len(question_token_to_idx))
            print('Get question_oie_subject_token_to_idx')
            print(len(question_oie_subject_token_to_idx))
            print('Get question_oie_relation_token_to_idx')
            print(len(question_oie_relation_token_to_idx))
            print('Get question_oie_object_token_to_idx')
            print(len(question_oie_object_token_to_idx))

            vocab = {
                'question_token_to_idx': question_token_to_idx,
                'answer_token_to_idx': answer_token_to_idx,
                'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
            }

            vocab_subject = {
                    'question_token_to_idx': question_oie_subject_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            vocab_relation = {
                    'question_token_to_idx': question_oie_relation_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            vocab_object = {
                    'question_token_to_idx': question_oie_object_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
            with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab, f, indent=4)
            print('Write into %s' % args.vocab_subject_json.format(args.question_type, args.question_type))
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_subject, f, indent=4)
            print('Write into %s' % args.vocab_relation_json.format(args.question_type, args.question_type))
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_relation, f, indent=4)
            print('Write into %s' % args.vocab_object_json.format(args.question_type, args.question_type))
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_object, f, indent=4)

            # split 10% of questions for evaluation
            split = int(0.9 * len(questions))
            train_questions = questions[:split]
            train_answers = answers[:split]
            train_video_names = video_names[:split]
            train_video_ids = video_ids[:split]

            val_questions = questions[split:]
            val_answers = answers[split:]
            val_video_names = video_names[split:]
            val_video_ids = video_ids[split:]

            openeded_encoding_data_oie_bert2(args, vocab, vocab_subject, vocab_relation, vocab_object, train_questions, train_video_names, train_video_ids, train_answers, mode='train')
            openeded_encoding_data_oie_bert2(args, vocab, vocab_subject, vocab_relation, vocab_object, val_questions, val_video_names, val_video_ids, val_answers, mode='val')
        else:
            print('Loading vocab')
            with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
                vocab = json.load(f)
            print('Loading oie_subject vocab')
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_subject = json.load(f)
            print('Loading oie_relation vocab')
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_relation = json.load(f)
            print('Loading oie_object vocab')
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_object = json.load(f)
            openeded_encoding_data_oie_bert2(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers, mode='test')

def process_questions_openended_oie_keybert(args):
    print('Loading data')
    if args.mode in ["train"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    kw_model = KeyBERT()  
    # roberta = TransformerDocumentEmbeddings('roberta-base')
    # kw_model = KeyBERT(model=roberta)
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])
    properties = {
        'openie.affinity_probability_cap': 1/3,
                        }
    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    with StanfordOpenIE(properties=properties) as client:
        if args.mode in ['train']:
            print('Building vocab')
            answer_cnt = {}

            if args.question_type == "frameqa":
                for i, answer in enumerate(answers):
                    answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

                answer_token_to_idx = {'<UNK>': 0}
                for token in answer_cnt:
                    answer_token_to_idx[token] = len(answer_token_to_idx)
                print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))
            elif args.question_type == 'count':
                answer_token_to_idx = {'<UNK>': 0}

            question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_subject_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_relation_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_object_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            for i, q in enumerate(questions):
                question = q.lower()[:-1]
                keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(2, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
                print(keywords)
                if keywords==[]:
                    q=question.split()
                    token_subject = ' '.join(q[0:round(len(q)/3)])
                    token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                    token_object = ' '.join(q[2*round(len(q)/3):len(q)])
                else:
                    token_subject = ''.join(keywords[0][0])
                    token_relation =''.join(keywords[1][0])
                    token_object = ''.join(keywords[2][0])
                for token in nltk.word_tokenize(token_subject):
                    if token not in question_oie_subject_token_to_idx:
                        question_oie_subject_token_to_idx[token] = len(question_oie_subject_token_to_idx)
                for token in nltk.word_tokenize(token_relation):
                    if token not in question_oie_relation_token_to_idx:
                        question_oie_relation_token_to_idx[token] = len(question_oie_relation_token_to_idx)
                for token in nltk.word_tokenize(token_object):
                    if token not in question_oie_object_token_to_idx:
                        question_oie_object_token_to_idx[token] = len(question_oie_object_token_to_idx)
                for token in nltk.word_tokenize(question):
                    if token not in question_token_to_idx:
                        question_token_to_idx[token] = len(question_token_to_idx)
            print('Get question_token_to_idx')
            print(len(question_token_to_idx))
            print('Get question_oie_subject_token_to_idx')
            print(len(question_oie_subject_token_to_idx))
            print('Get question_oie_relation_token_to_idx')
            print(len(question_oie_relation_token_to_idx))
            print('Get question_oie_object_token_to_idx')
            print(len(question_oie_object_token_to_idx))

            vocab = {
                'question_token_to_idx': question_token_to_idx,
                'answer_token_to_idx': answer_token_to_idx,
                'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
            }

            vocab_subject = {
                    'question_token_to_idx': question_oie_subject_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            vocab_relation = {
                    'question_token_to_idx': question_oie_relation_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            vocab_object = {
                    'question_token_to_idx': question_oie_object_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
            with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab, f, indent=4)
            print('Write into %s' % args.vocab_subject_json.format(args.question_type, args.question_type))
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_subject, f, indent=4)
            print('Write into %s' % args.vocab_relation_json.format(args.question_type, args.question_type))
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_relation, f, indent=4)
            print('Write into %s' % args.vocab_object_json.format(args.question_type, args.question_type))
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_object, f, indent=4)

            # split 10% of questions for evaluation
            split = int(0.9 * len(questions))
            train_questions = questions[:split]
            train_answers = answers[:split]
            train_video_names = video_names[:split]
            train_video_ids = video_ids[:split]

            val_questions = questions[split:]
            val_answers = answers[split:]
            val_video_names = video_names[split:]
            val_video_ids = video_ids[split:]

            openeded_encoding_data_oie_bert2(args, vocab, vocab_subject, vocab_relation, vocab_object, train_questions, train_video_names, train_video_ids, train_answers, mode='train')
            openeded_encoding_data_oie_bert2(args, vocab, vocab_subject, vocab_relation, vocab_object, val_questions, val_video_names, val_video_ids, val_answers, mode='val')
        else:
            print('Loading vocab')
            with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
                vocab = json.load(f)
            print('Loading oie_subject vocab')
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_subject = json.load(f)
            print('Loading oie_relation vocab')
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_relation = json.load(f)
            print('Loading oie_object vocab')
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_object = json.load(f)
            openeded_encoding_data_oie_bert2(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers, mode='test')

def process_questions_openended_oie(args):
    print('Loading data')
    if args.mode in ["train"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    with StanfordOpenIE(properties=properties) as client:
        if args.mode in ['train']:
            print('Building vocab')
            answer_cnt = {}

            if args.question_type == "frameqa":
                for i, answer in enumerate(answers):
                    answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

                answer_token_to_idx = {'<UNK>': 0}
                for token in answer_cnt:
                    answer_token_to_idx[token] = len(answer_token_to_idx)
                print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))
            elif args.question_type == 'count':
                answer_token_to_idx = {'<UNK>': 0}

            question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_subject_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_relation_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_object_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            for i, q in enumerate(questions):
                question = q.lower()[:-1]
                if client.annotate(question)==[]:
                    q=question.split()
                    token_subject = ' '.join(q[0:round(len(q)/3)])
                    token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                    token_object = ' '.join(q[2*round(len(q)/3):len(q)])
                else:
                    token_subject = client.annotate(question)[0]['subject']
                    token_relation = client.annotate(question)[0]['relation']
                    token_object = client.annotate(question)[0]['object']
                for token in nltk.word_tokenize(token_subject):
                    if token not in question_oie_subject_token_to_idx:
                        question_oie_subject_token_to_idx[token] = len(question_oie_subject_token_to_idx)
                for token in nltk.word_tokenize(token_relation):
                    if token not in question_oie_relation_token_to_idx:
                        question_oie_relation_token_to_idx[token] = len(question_oie_relation_token_to_idx)
                for token in nltk.word_tokenize(token_object):
                    if token not in question_oie_object_token_to_idx:
                        question_oie_object_token_to_idx[token] = len(question_oie_object_token_to_idx)
                for token in nltk.word_tokenize(question):
                    if token not in question_token_to_idx:
                        question_token_to_idx[token] = len(question_token_to_idx)
            print('Get question_token_to_idx')
            print(len(question_token_to_idx))
            print('Get question_oie_subject_token_to_idx')
            print(len(question_oie_subject_token_to_idx))
            print('Get question_oie_relation_token_to_idx')
            print(len(question_oie_relation_token_to_idx))
            print('Get question_oie_object_token_to_idx')
            print(len(question_oie_object_token_to_idx))

            vocab = {
                'question_token_to_idx': question_token_to_idx,
                'answer_token_to_idx': answer_token_to_idx,
                'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
            }

            vocab_subject = {
                    'question_token_to_idx': question_oie_subject_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            vocab_relation = {
                    'question_token_to_idx': question_oie_relation_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            vocab_object = {
                    'question_token_to_idx': question_oie_object_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
            with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab, f, indent=4)
            print('Write into %s' % args.vocab_subject_json.format(args.question_type, args.question_type))
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_subject, f, indent=4)
            print('Write into %s' % args.vocab_relation_json.format(args.question_type, args.question_type))
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_relation, f, indent=4)
            print('Write into %s' % args.vocab_object_json.format(args.question_type, args.question_type))
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_object, f, indent=4)

            # split 10% of questions for evaluation
            split = int(0.9 * len(questions))
            train_questions = questions[:split]
            train_answers = answers[:split]
            train_video_names = video_names[:split]
            train_video_ids = video_ids[:split]

            val_questions = questions[split:]
            val_answers = answers[split:]
            val_video_names = video_names[split:]
            val_video_ids = video_ids[split:]

            openeded_encoding_data_oie(args, vocab, vocab_subject, vocab_relation, vocab_object, train_questions, train_video_names, train_video_ids, train_answers, mode='train')
            openeded_encoding_data_oie(args, vocab, vocab_subject, vocab_relation, vocab_object, val_questions, val_video_names, val_video_ids, val_answers, mode='val')
        else:
            print('Loading vocab')
            with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
                vocab = json.load(f)
            print('Loading oie_subject vocab')
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_subject = json.load(f)
            print('Loading oie_relation vocab')
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_relation = json.load(f)
            print('Loading oie_object vocab')
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_object = json.load(f)
            openeded_encoding_data_oie(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers, mode='test')

def process_questions_openended_oie_bert3(args):
    print('Loading data')
    if args.mode in ["train"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    with StanfordOpenIE(properties=properties) as client:
        if args.mode in ['train']:
            print('Building vocab')
            answer_cnt = {}

            if args.question_type == "frameqa":
                for i, answer in enumerate(answers):
                    answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

                answer_token_to_idx = {'<UNK>': 0}
                for token in answer_cnt:
                    answer_token_to_idx[token] = len(answer_token_to_idx)
                print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))
            elif args.question_type == 'count':
                answer_token_to_idx = {'<UNK>': 0}

            question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_subject_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_relation_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_object_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            for i, q in enumerate(questions):
                question = q.lower()[:-1]
                if client.annotate(question)==[]:
                    q=question.split()
                    token_subject = ' '.join(q[0:round(len(q)/3)])
                    token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                    token_object = ' '.join(q[2*round(len(q)/3):len(q)])
                else:
                    token_subject = client.annotate(question)[0]['subject']
                    token_relation = client.annotate(question)[0]['relation']
                    token_object = client.annotate(question)[0]['object']
                for token in nltk.word_tokenize(token_subject):
                    if token not in question_oie_subject_token_to_idx:
                        question_oie_subject_token_to_idx[token] = len(question_oie_subject_token_to_idx)
                for token in nltk.word_tokenize(token_relation):
                    if token not in question_oie_relation_token_to_idx:
                        question_oie_relation_token_to_idx[token] = len(question_oie_relation_token_to_idx)
                for token in nltk.word_tokenize(token_object):
                    if token not in question_oie_object_token_to_idx:
                        question_oie_object_token_to_idx[token] = len(question_oie_object_token_to_idx)
                for token in nltk.word_tokenize(question):
                    if token not in question_token_to_idx:
                        question_token_to_idx[token] = len(question_token_to_idx)
            print('Get question_token_to_idx')
            print(len(question_token_to_idx))
            print('Get question_oie_subject_token_to_idx')
            print(len(question_oie_subject_token_to_idx))
            print('Get question_oie_relation_token_to_idx')
            print(len(question_oie_relation_token_to_idx))
            print('Get question_oie_object_token_to_idx')
            print(len(question_oie_object_token_to_idx))

            vocab = {
                'question_token_to_idx': question_token_to_idx,
                'answer_token_to_idx': answer_token_to_idx,
                'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
            }

            vocab_subject = {
                    'question_token_to_idx': question_oie_subject_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            vocab_relation = {
                    'question_token_to_idx': question_oie_relation_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            vocab_object = {
                    'question_token_to_idx': question_oie_object_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
            with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab, f, indent=4)
            print('Write into %s' % args.vocab_subject_json.format(args.question_type, args.question_type))
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_subject, f, indent=4)
            print('Write into %s' % args.vocab_relation_json.format(args.question_type, args.question_type))
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_relation, f, indent=4)
            print('Write into %s' % args.vocab_object_json.format(args.question_type, args.question_type))
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_object, f, indent=4)

            # split 10% of questions for evaluation
            split = int(0.9 * len(questions))
            train_questions = questions[:split]
            train_answers = answers[:split]
            train_video_names = video_names[:split]
            train_video_ids = video_ids[:split]

            val_questions = questions[split:]
            val_answers = answers[split:]
            val_video_names = video_names[split:]
            val_video_ids = video_ids[split:]

            openeded_encoding_data_oie_bert3(args, vocab, vocab_subject, vocab_relation, vocab_object, train_questions, train_video_names, train_video_ids, train_answers, mode='train')
            openeded_encoding_data_oie_bert3(args, vocab, vocab_subject, vocab_relation, vocab_object, val_questions, val_video_names, val_video_ids, val_answers, mode='val')
        else:
            print('Loading vocab')
            with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
                vocab = json.load(f)
            print('Loading oie_subject vocab')
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_subject = json.load(f)
            print('Loading oie_relation vocab')
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_relation = json.load(f)
            print('Loading oie_object vocab')
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_object = json.load(f)
            openeded_encoding_data_oie_bert3(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers, mode='test')

def process_questions_mulchoices(args):
    print('Loading data')
    if args.mode in ["train", "val"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])
    ans_candidates = np.asarray(
        [csv_data['a1'], csv_data['a2'], csv_data['a3'], csv_data['a4'], csv_data['a5']])
    ans_candidates = ans_candidates.transpose()
    print(ans_candidates.shape)
    # ans_candidates: (num_ques, 5)
    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        print('Building vocab')

        answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
        question_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for candidates in ans_candidates:
            for ans in candidates:
                ans = ans.lower()
                for token in nltk.word_tokenize(ans):
                    if token not in answer_token_to_idx:
                        answer_token_to_idx[token] = len(answer_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for i, q in enumerate(questions):
            question = q.lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)

        print('Get question_token_to_idx')
        print(len(question_token_to_idx))
        print('Get question_answer_token_to_idx')
        print(len(question_answer_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'question_answer_token_to_idx': question_answer_token_to_idx,
        }

        print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
        with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
            json.dump(vocab, f, indent=4)

        # split 10% of questions for evaluation
        split = int(0.9 * len(questions))
        train_questions = questions[:split]
        train_answers = answers[:split]
        train_video_names = video_names[:split]
        train_video_ids = video_ids[:split]
        train_ans_candidates = ans_candidates[:split, :]

        val_questions = questions[split:]
        val_answers = answers[split:]
        val_video_names = video_names[split:]
        val_video_ids = video_ids[split:]
        val_ans_candidates = ans_candidates[split:, :]

        multichoice_encoding_data(args, vocab, train_questions, train_video_names, train_video_ids, train_answers, train_ans_candidates, mode='train')
        multichoice_encoding_data(args, vocab, val_questions, val_video_names, val_video_ids, val_answers,
                                  val_ans_candidates, mode='val')
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
            vocab = json.load(f)
        multichoice_encoding_data(args, vocab, questions, video_names, video_ids, answers,
                                  ans_candidates, mode='test')

def process_questions_mulchoices_oie_keybert(args):
    print('Loading data')
    if args.mode in ["train", "val"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    kw_model = KeyBERT()  
    # roberta = TransformerDocumentEmbeddings('roberta-base')
    # kw_model = KeyBERT(model=roberta)
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])
    ans_candidates = np.asarray(
        [csv_data['a1'], csv_data['a2'], csv_data['a3'], csv_data['a4'], csv_data['a5']])
    ans_candidates = ans_candidates.transpose()
    print(ans_candidates.shape)
    # ans_candidates: (num_ques, 5)
    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    with StanfordOpenIE(properties=properties) as client:
        if args.mode in ['train']:
            print('Building vocab')

            answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
            question_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_subject_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_relation_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_object_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            for candidates in ans_candidates:
                for ans in candidates:
                    ans = ans.lower()
                    for token in nltk.word_tokenize(ans):
                        if token not in answer_token_to_idx:
                            answer_token_to_idx[token] = len(answer_token_to_idx)
                        if token not in question_answer_token_to_idx:
                            question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
            print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

            question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_subject_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_relation_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_object_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            for i, q in enumerate(questions):
                question = q.lower()[:-1]
                keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(2, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
                print(keywords)
                if keywords==[]:
                    q=question.split()
                    token_subject = ' '.join(q[0:round(len(q)/3)])
                    token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                    token_object = ' '.join(q[2*round(len(q)/3):len(q)])
                else:
                    token_subject = ''.join(keywords[0][0])
                    token_relation =''.join(keywords[1][0])
                    token_object = ''.join(keywords[2][0])

                for token in nltk.word_tokenize(token_subject):
                    if token not in question_oie_subject_token_to_idx:
                        question_oie_subject_token_to_idx[token] = len(question_oie_subject_token_to_idx)
                    if token not in question_subject_answer_token_to_idx:
                        question_subject_answer_token_to_idx[token] = len(question_subject_answer_token_to_idx)
    
                for token in nltk.word_tokenize(token_relation):
                    if token not in question_oie_relation_token_to_idx:
                        question_oie_relation_token_to_idx[token] = len(question_oie_relation_token_to_idx)
                    if token not in question_relation_answer_token_to_idx:
                        question_relation_answer_token_to_idx[token] = len(question_relation_answer_token_to_idx)

                for token in nltk.word_tokenize(token_object):
                    if token not in question_oie_object_token_to_idx:
                        question_oie_object_token_to_idx[token] = len(question_oie_object_token_to_idx)
                    if token not in question_object_answer_token_to_idx:
                        question_object_answer_token_to_idx[token] = len(question_object_answer_token_to_idx)

                for token in nltk.word_tokenize(question):
                    if token not in question_token_to_idx:
                        question_token_to_idx[token] = len(question_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(question_answer_token_to_idx)


            print('Get question_oie_subject_token_to_idx')
            print(len(question_oie_subject_token_to_idx))
            print('Get question_subject_answer_token_to_idx')
            print(len(question_subject_answer_token_to_idx))

            print('Get question_oie_relation_token_to_idx')
            print(len(question_oie_relation_token_to_idx))
            print('Get question_relation_answer_token_to_idx')
            print(len(question_relation_answer_token_to_idx))

            print('Get question_oie_object_token_to_idx')
            print(len(question_oie_object_token_to_idx))
            print('Get question_object_answer_token_to_idx')
            print(len(question_object_answer_token_to_idx))

            print('Get question_token_to_idx')
            print(len(question_token_to_idx))
            print('Get question_answer_token_to_idx')
            print(len(question_answer_token_to_idx))

            vocab = {
                'question_token_to_idx': question_token_to_idx,
                'answer_token_to_idx': answer_token_to_idx,
                'question_answer_token_to_idx': question_answer_token_to_idx,
            }

            vocab_subject = {
                    'question_token_to_idx': question_oie_subject_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': question_subject_answer_token_to_idx,
                }

            vocab_relation = {
                    'question_token_to_idx': question_oie_relation_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': question_relation_answer_token_to_idx,
                }

            vocab_object = {
                    'question_token_to_idx': question_oie_object_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': question_object_answer_token_to_idx,
                }
            print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
            with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab, f, indent=4)
            print('Write into %s' % args.vocab_subject_json.format(args.question_type, args.question_type))
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_subject, f, indent=4)
            print('Write into %s' % args.vocab_relation_json.format(args.question_type, args.question_type))
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_relation, f, indent=4)
            print('Write into %s' % args.vocab_object_json.format(args.question_type, args.question_type))
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_object, f, indent=4)

            # split 10% of questions for evaluation
            split = int(0.9 * len(questions))
            train_questions = questions[:split]
            train_answers = answers[:split]
            train_video_names = video_names[:split]
            train_video_ids = video_ids[:split]
            train_ans_candidates = ans_candidates[:split, :]

            val_questions = questions[split:]
            val_answers = answers[split:]
            val_video_names = video_names[split:]
            val_video_ids = video_ids[split:]
            val_ans_candidates = ans_candidates[split:, :]

            multichoice_encoding_data_oie_keybert(args, vocab, vocab_subject, vocab_relation, vocab_object, train_questions, train_video_names, train_video_ids, train_answers, train_ans_candidates, mode='train')
            multichoice_encoding_data_oie_keybert(args, vocab, vocab_subject, vocab_relation, vocab_object, val_questions, val_video_names, val_video_ids, val_answers,
                                    val_ans_candidates, mode='val')
        else:
            print('Loading vocab')
            with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
                vocab = json.load(f)
            print('Loading oie_subject vocab')
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_subject = json.load(f)
            print('Loading oie_relation vocab')
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_relation = json.load(f)
            print('Loading oie_object vocab')
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_object = json.load(f)
            multichoice_encoding_data_oie_keybert(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers,
                                    ans_candidates, mode='test')

def process_questions_mulchoices_oie_keybert2(args):
    print('Loading data')
    if args.mode in ["train", "val"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    kw_model = KeyBERT()  
    # roberta = TransformerDocumentEmbeddings('roberta-base')
    # kw_model = KeyBERT(model=roberta)
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])
    ans_candidates = np.asarray(
        [csv_data['a1'], csv_data['a2'], csv_data['a3'], csv_data['a4'], csv_data['a5']])
    ans_candidates = ans_candidates.transpose()
    print(ans_candidates.shape)
    # ans_candidates: (num_ques, 5)
    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    with StanfordOpenIE(properties=properties) as client:
        if args.mode in ['train']:
            print('Building vocab')

            answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
            question_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_subject_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_relation_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_object_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            for candidates in ans_candidates:
                for ans in candidates:
                    ans = ans.lower()
                    for token in nltk.word_tokenize(ans):
                        if token not in answer_token_to_idx:
                            answer_token_to_idx[token] = len(answer_token_to_idx)
                        if token not in question_answer_token_to_idx:
                            question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
            print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

            question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_subject_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_relation_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_object_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            for i, q in enumerate(questions):
                question = q.lower()[:-1]
                #keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(2, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
                #print(keywords)
                # if keywords==[]:
                if client.annotate(question)==[]:
                    q=question.split()
                    token_subject = ' '.join(q[0:round(len(q)/3)])
                    token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                    token_object = ' '.join(q[2*round(len(q)/3):len(q)])
                else:
                    # token_subject = ''.join(keywords[0][0])
                    # token_relation =''.join(keywords[1][0])
                    # token_object = ''.join(keywords[2][0])
                    token_subject = client.annotate(question)[0]['subject']
                    token_relation = client.annotate(question)[0]['relation']
                    token_object = client.annotate(question)[0]['object']

                for token in nltk.word_tokenize(token_subject):
                    if token not in question_oie_subject_token_to_idx:
                        question_oie_subject_token_to_idx[token] = len(question_oie_subject_token_to_idx)
                    if token not in question_subject_answer_token_to_idx:
                        question_subject_answer_token_to_idx[token] = len(question_subject_answer_token_to_idx)
    
                for token in nltk.word_tokenize(token_relation):
                    if token not in question_oie_relation_token_to_idx:
                        question_oie_relation_token_to_idx[token] = len(question_oie_relation_token_to_idx)
                    if token not in question_relation_answer_token_to_idx:
                        question_relation_answer_token_to_idx[token] = len(question_relation_answer_token_to_idx)

                for token in nltk.word_tokenize(token_object):
                    if token not in question_oie_object_token_to_idx:
                        question_oie_object_token_to_idx[token] = len(question_oie_object_token_to_idx)
                    if token not in question_object_answer_token_to_idx:
                        question_object_answer_token_to_idx[token] = len(question_object_answer_token_to_idx)

                for token in nltk.word_tokenize(question):
                    if token not in question_token_to_idx:
                        question_token_to_idx[token] = len(question_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(question_answer_token_to_idx)


            print('Get question_oie_subject_token_to_idx')
            print(len(question_oie_subject_token_to_idx))
            print('Get question_subject_answer_token_to_idx')
            print(len(question_subject_answer_token_to_idx))

            print('Get question_oie_relation_token_to_idx')
            print(len(question_oie_relation_token_to_idx))
            print('Get question_relation_answer_token_to_idx')
            print(len(question_relation_answer_token_to_idx))

            print('Get question_oie_object_token_to_idx')
            print(len(question_oie_object_token_to_idx))
            print('Get question_object_answer_token_to_idx')
            print(len(question_object_answer_token_to_idx))

            print('Get question_token_to_idx')
            print(len(question_token_to_idx))
            print('Get question_answer_token_to_idx')
            print(len(question_answer_token_to_idx))

            vocab = {
                'question_token_to_idx': question_token_to_idx,
                'answer_token_to_idx': answer_token_to_idx,
                'question_answer_token_to_idx': question_answer_token_to_idx,
            }

            vocab_subject = {
                    'question_token_to_idx': question_oie_subject_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': question_subject_answer_token_to_idx,
                }

            vocab_relation = {
                    'question_token_to_idx': question_oie_relation_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': question_relation_answer_token_to_idx,
                }

            vocab_object = {
                    'question_token_to_idx': question_oie_object_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': question_object_answer_token_to_idx,
                }
            print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
            with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab, f, indent=4)
            print('Write into %s' % args.vocab_subject_json.format(args.question_type, args.question_type))
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_subject, f, indent=4)
            print('Write into %s' % args.vocab_relation_json.format(args.question_type, args.question_type))
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_relation, f, indent=4)
            print('Write into %s' % args.vocab_object_json.format(args.question_type, args.question_type))
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_object, f, indent=4)

            # split 10% of questions for evaluation
            split = int(0.9 * len(questions))
            train_questions = questions[:split]
            train_answers = answers[:split]
            train_video_names = video_names[:split]
            train_video_ids = video_ids[:split]
            train_ans_candidates = ans_candidates[:split, :]

            val_questions = questions[split:]
            val_answers = answers[split:]
            val_video_names = video_names[split:]
            val_video_ids = video_ids[split:]
            val_ans_candidates = ans_candidates[split:, :]

            multichoice_encoding_data_oie_bert2(args, vocab, vocab_subject, vocab_relation, vocab_object, train_questions, train_video_names, train_video_ids, train_answers, train_ans_candidates, mode='train')
            multichoice_encoding_data_oie_bert2(args, vocab, vocab_subject, vocab_relation, vocab_object, val_questions, val_video_names, val_video_ids, val_answers,
                                    val_ans_candidates, mode='val')
        else:
            print('Loading vocab')
            with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
                vocab = json.load(f)
            print('Loading oie_subject vocab')
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_subject = json.load(f)
            print('Loading oie_relation vocab')
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_relation = json.load(f)
            print('Loading oie_object vocab')
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_object = json.load(f)
            multichoice_encoding_data_oie_bert2(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers,
                                    ans_candidates, mode='test')


def process_questions_mulchoices_oie(args):
    print('Loading data')
    if args.mode in ["train", "val"]:
        csv_data = pd.read_csv(args.annotation_file.format("Train", args.question_type), delimiter='\t')
    else:
        csv_data = pd.read_csv(args.annotation_file.format("Test", args.question_type), delimiter='\t')
    csv_data = csv_data.iloc[np.random.permutation(len(csv_data))]
    questions = list(csv_data['question'])
    answers = list(csv_data['answer'])
    video_names = list(csv_data['gif_name'])
    video_ids = list(csv_data['key'])
    ans_candidates = np.asarray(
        [csv_data['a1'], csv_data['a2'], csv_data['a3'], csv_data['a4'], csv_data['a5']])
    ans_candidates = ans_candidates.transpose()
    print(ans_candidates.shape)
    # ans_candidates: (num_ques, 5)
    print('number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    with StanfordOpenIE(properties=properties) as client:
        if args.mode in ['train']:
            print('Building vocab')

            answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
            question_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_subject_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_relation_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_object_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            for candidates in ans_candidates:
                for ans in candidates:
                    ans = ans.lower()
                    for token in nltk.word_tokenize(ans):
                        if token not in answer_token_to_idx:
                            answer_token_to_idx[token] = len(answer_token_to_idx)
                        if token not in question_answer_token_to_idx:
                            question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
            print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

            question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_subject_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_relation_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_object_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            for i, q in enumerate(questions):
                question = q.lower()[:-1]
                if client.annotate(question)==[]:
                    q=question.split()
                    token_subject = ' '.join(q[0:round(len(q)/3)])
                    token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                    token_object = ' '.join(q[2*round(len(q)/3):len(q)])
                else:
                    token_subject = client.annotate(question)[0]['subject']
                    token_relation = client.annotate(question)[0]['relation']
                    token_object = client.annotate(question)[0]['object']

                for token in nltk.word_tokenize(token_subject):
                    if token not in question_oie_subject_token_to_idx:
                        question_oie_subject_token_to_idx[token] = len(question_oie_subject_token_to_idx)
                    if token not in question_subject_answer_token_to_idx:
                        question_subject_answer_token_to_idx[token] = len(question_subject_answer_token_to_idx)
    
                for token in nltk.word_tokenize(token_relation):
                    if token not in question_oie_relation_token_to_idx:
                        question_oie_relation_token_to_idx[token] = len(question_oie_relation_token_to_idx)
                    if token not in question_relation_answer_token_to_idx:
                        question_relation_answer_token_to_idx[token] = len(question_relation_answer_token_to_idx)

                for token in nltk.word_tokenize(token_object):
                    if token not in question_oie_object_token_to_idx:
                        question_oie_object_token_to_idx[token] = len(question_oie_object_token_to_idx)
                    if token not in question_object_answer_token_to_idx:
                        question_object_answer_token_to_idx[token] = len(question_object_answer_token_to_idx)

                for token in nltk.word_tokenize(question):
                    if token not in question_token_to_idx:
                        question_token_to_idx[token] = len(question_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(question_answer_token_to_idx)


            print('Get question_oie_subject_token_to_idx')
            print(len(question_oie_subject_token_to_idx))
            print('Get question_subject_answer_token_to_idx')
            print(len(question_subject_answer_token_to_idx))

            print('Get question_oie_relation_token_to_idx')
            print(len(question_oie_relation_token_to_idx))
            print('Get question_relation_answer_token_to_idx')
            print(len(question_relation_answer_token_to_idx))

            print('Get question_oie_object_token_to_idx')
            print(len(question_oie_object_token_to_idx))
            print('Get question_object_answer_token_to_idx')
            print(len(question_object_answer_token_to_idx))

            print('Get question_token_to_idx')
            print(len(question_token_to_idx))
            print('Get question_answer_token_to_idx')
            print(len(question_answer_token_to_idx))

            vocab = {
                'question_token_to_idx': question_token_to_idx,
                'answer_token_to_idx': answer_token_to_idx,
                'question_answer_token_to_idx': question_answer_token_to_idx,
            }

            vocab_subject = {
                    'question_token_to_idx': question_oie_subject_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': question_subject_answer_token_to_idx,
                }

            vocab_relation = {
                    'question_token_to_idx': question_oie_relation_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': question_relation_answer_token_to_idx,
                }

            vocab_object = {
                    'question_token_to_idx': question_oie_object_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': question_object_answer_token_to_idx,
                }
            print('Write into %s' % args.vocab_json.format(args.question_type, args.question_type))
            with open(args.vocab_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab, f, indent=4)
            print('Write into %s' % args.vocab_subject_json.format(args.question_type, args.question_type))
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_subject, f, indent=4)
            print('Write into %s' % args.vocab_relation_json.format(args.question_type, args.question_type))
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_relation, f, indent=4)
            print('Write into %s' % args.vocab_object_json.format(args.question_type, args.question_type))
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'w') as f:
                json.dump(vocab_object, f, indent=4)

            # split 10% of questions for evaluation
            split = int(0.9 * len(questions))
            train_questions = questions[:split]
            train_answers = answers[:split]
            train_video_names = video_names[:split]
            train_video_ids = video_ids[:split]
            train_ans_candidates = ans_candidates[:split, :]

            val_questions = questions[split:]
            val_answers = answers[split:]
            val_video_names = video_names[split:]
            val_video_ids = video_ids[split:]
            val_ans_candidates = ans_candidates[split:, :]

            multichoice_encoding_data_oie(args, vocab, vocab_subject, vocab_relation, vocab_object, train_questions, train_video_names, train_video_ids, train_answers, train_ans_candidates, mode='train')
            multichoice_encoding_data_oie(args, vocab, vocab_subject, vocab_relation, vocab_object, val_questions, val_video_names, val_video_ids, val_answers,
                                    val_ans_candidates, mode='val')
        else:
            print('Loading vocab')
            with open(args.vocab_json.format(args.question_type, args.question_type), 'r') as f:
                vocab = json.load(f)
            print('Loading oie_subject vocab')
            with open(args.vocab_subject_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_subject = json.load(f)
            print('Loading oie_relation vocab')
            with open(args.vocab_relation_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_relation = json.load(f)
            print('Loading oie_object vocab')
            with open(args.vocab_object_json.format(args.question_type, args.question_type), 'r') as f:
                vocab_object = json.load(f)
            multichoice_encoding_data_oie(args, vocab, vocab_subject, vocab_relation, vocab_object, questions, video_names, video_ids, answers,
                                    ans_candidates, mode='test')