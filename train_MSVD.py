import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import logging
from termcolor import colored
from torch._six import inf
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from DataLoader import  VideoQADataLoader_oie_bert_opendended
from utils import todevice
from validate_msvd import validate

import model.VLCIR_msvd_bertv2 as VLCIR
from utils import todevice

from config import cfg, cfg_from_file
lctime = time.localtime()
lctime = time.strftime("%Y-%m-%d_%A_%H:%M:%S",lctime)
def train_oie(cfg):
    logging.info("Create train_loader and val_loader.........")
    train_loader_kwargs = {
        'question_type': cfg.dataset.question_type,
        'question_pt': cfg.dataset.train_question_pt,
        'question_subject_pt': cfg.dataset.train_question_subject_pt,
        'question_relation_pt': cfg.dataset.train_question_relation_pt,
        'question_object_pt': cfg.dataset.train_question_object_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'vocab_subject_json': cfg.dataset.vocab_subject_json,
        'vocab_relation_json': cfg.dataset.vocab_relation_json,
        'vocab_object_json': cfg.dataset.vocab_object_json,        
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'appearance_dict': cfg.dataset.appearance_dict,
        'motion_dict': cfg.dataset.motion_dict,
        'train_num': cfg.train.train_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': True
    }
    train_loader = VideoQADataLoader_oie_bert_opendended(**train_loader_kwargs)
    logging.info("number of train instances: {}".format(len(train_loader.dataset)))
    if cfg.val.flag:
        val_loader_kwargs = {
            'question_type': cfg.dataset.question_type,
            'question_pt': cfg.dataset.val_question_pt,
            'question_subject_pt': cfg.dataset.val_question_subject_pt,
            'question_relation_pt': cfg.dataset.val_question_relation_pt,
            'question_object_pt': cfg.dataset.val_question_object_pt,
            'vocab_json': cfg.dataset.vocab_json,
            'vocab_subject_json': cfg.dataset.vocab_subject_json,
            'vocab_relation_json': cfg.dataset.vocab_relation_json,
            'vocab_object_json': cfg.dataset.vocab_object_json,  
            'appearance_feat': cfg.dataset.appearance_feat,
            'motion_feat': cfg.dataset.motion_feat,
            'appearance_dict': cfg.dataset.appearance_dict,
            'motion_dict': cfg.dataset.motion_dict,
            'val_num': cfg.val.val_num,
            'batch_size': cfg.train.batch_size,
            'num_workers': cfg.num_workers,
            'shuffle': False
        }
        val_loader = VideoQADataLoader_oie_bert_opendended(**val_loader_kwargs)
        logging.info("number of val instances: {}".format(len(val_loader.dataset)))

    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {
        'motion_dim': cfg.train.motion_dim,
        'appearance_dim': cfg.train.appearance_dim,
        'module_dim': cfg.train.module_dim,
        'word_dim': cfg.train.word_dim,
        # 'k_max_frame_level': cfg.train.k_max_frame_level,
        # 'k_max_clip_level': cfg.train.k_max_clip_level,
        # 'spl_resolution': cfg.train.spl_resolution,
        'vocab': train_loader.vocab,
        'vocab_subject': train_loader.vocab_subject,
        'vocab_relation': train_loader.vocab_relation,
        'vocab_object': train_loader.vocab_object,
        'question_type': cfg.dataset.question_type
    }
    model_kwargs_tosave = {k: v for k, v in model_kwargs.items() if k != 'vocab' or k != 'vocab_subject' or k != 'vocab_relation' or k != 'vocab_object'} 
    model = VLCIR.VLCIR(**model_kwargs).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('num of params: {}'.format(pytorch_total_params))
    logging.info(model)

    if cfg.train.glove:
        logging.info('load glove vectors')
        train_loader.glove_matrix = torch.FloatTensor(train_loader.glove_matrix).to(device)
        train_loader.glove_matrix_subject = torch.FloatTensor(train_loader.glove_matrix_subject).to(device)
        train_loader.glove_matrix_relation = torch.FloatTensor(train_loader.glove_matrix_relation).to(device)
        train_loader.glove_matrix_object = torch.FloatTensor(train_loader.glove_matrix_object).to(device)
        with torch.no_grad():
            model.linguistic_input_unit.encoder_embed.weight.set_(train_loader.glove_matrix)
            model.linguistic_subject_input_unit.encoder_embed.weight.set_(train_loader.glove_matrix_subject)
            model.linguistic_relation_input_unit.encoder_embed.weight.set_(train_loader.glove_matrix_relation)
            model.linguistic_object_input_unit.encoder_embed.weight.set_(train_loader.glove_matrix_object)
            model.linguistic_input_unit_do.encoder_embed.weight.set_(train_loader.glove_matrix)
            model.linguistic_subject_input_unit_do.encoder_embed.weight.set_(train_loader.glove_matrix_subject)
            model.linguistic_relation_input_unit_do.encoder_embed.weight.set_(train_loader.glove_matrix_relation)
            model.linguistic_object_input_unit_do.encoder_embed.weight.set_(train_loader.glove_matrix_object)
    if torch.cuda.device_count() > 1 and cfg.multi_gpus:
        model = model.cuda()
        logging.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1])

    optimizer = optim.Adam(model.parameters(), cfg.train.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
    if cfg.dataset.question_type in ['none', 'frameqa']:
        criterion = nn.CrossEntropyLoss().cuda()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=5,factor=0.5,verbose=True)
    elif cfg.dataset.question_type in ['count']:
        criterion = nn.MSELoss().cuda()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,factor=0.5,verbose=True)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=5,factor=0.5,verbose=True)
    start_epoch = 0
    if cfg.dataset.name == 'msvd-qa' or cfg.dataset.name == 'msrvtt-qa':
        best_val = 0.
        best_what = 0.
        best_how = 0.
        best_when = 0.
        best_who = 0.
        best_where = 0.    
    if cfg.dataset.question_type == 'count':
        best_val = 100.0
    else:
        best_val = 0
    if cfg.train.restore:
        print("Restore checkpoint and optimizer...")
        ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
        ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
    if cfg.dataset.question_type in ['frameqa', 'none']:
        criterion = nn.CrossEntropyLoss().to(device)
    elif cfg.dataset.question_type == 'count':
        criterion = nn.MSELoss().to(device)
    logging.info("Start training........")
    for epoch in range(start_epoch, cfg.train.max_epochs):
        logging.info('>>>>>> epoch {epoch} <<<<<<'.format(epoch=colored("{}".format(epoch), "green", attrs=["bold"])))
        model.train()
        total_acc, count = 0, 0
        batch_mse_sum = 0.0
        total_loss, avg_loss = 0.0, 0.0
        avg_loss = 0
        train_accuracy = 0
        for i, batch in enumerate(iter(train_loader)):
            progress = epoch + i / len(train_loader)
            _, _, answers, *batch_input = [todevice(x, device) for x in batch]
            answers = answers.cuda().squeeze()
            batch_size = answers.size(0)
            optimizer.zero_grad()
            logits = model(*batch_input)
            if cfg.dataset.question_type in ['action', 'transition']:
                batch_agg = np.concatenate(np.tile(np.arange(batch_size).reshape([batch_size, 1]),
                                                   [1, 5])) * 5  # [0, 0, 0, 0, 0, 5, 5, 5, 5, 1, ...]
                answers_agg = tile(answers, 0, 5)
                loss = torch.max(torch.tensor(0.0).cuda(),
                                 1.0 + logits - logits[answers_agg + torch.from_numpy(batch_agg).cuda()])
                loss = loss.sum()
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (i + 1)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                optimizer.step()
                preds = torch.argmax(logits.view(batch_size, 5), dim=1)
                aggreeings = (preds == answers)
            elif cfg.dataset.question_type == 'count':
                answers = answers.unsqueeze(-1)
                loss = criterion(logits, answers.float())
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (i + 1)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                optimizer.step()
                preds = (logits + 0.5).long().clamp(min=1, max=10)
                batch_mse = (preds - answers) ** 2
            else:
                loss = criterion(logits, answers)
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (i + 1)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                optimizer.step()
                aggreeings = batch_accuracy(logits, answers)

            if cfg.dataset.question_type == 'count':
                batch_avg_mse = batch_mse.sum().item() / answers.size(0)
                batch_mse_sum += batch_mse.sum().item()
                count += answers.size(0)
                avg_mse = batch_mse_sum / count
                sys.stdout.write(
                    "\rProgress = {progress}   ce_loss = {ce_loss}   avg_loss = {avg_loss}    train_mse = {train_mse}    avg_mse = {avg_mse}".format(
                        progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                        ce_loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                        avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                        train_mse=colored("{:.4f}".format(batch_avg_mse), "blue",
                                          attrs=['bold']),
                        avg_mse=colored("{:.4f}".format(avg_mse), "red", attrs=['bold'])))
                sys.stdout.flush()
            else:
                total_acc += aggreeings.sum().item()
                count += answers.size(0)
                train_accuracy = total_acc / count
                sys.stdout.write(
                    "\rProgress = {progress}   ce_loss = {ce_loss}   avg_loss = {avg_loss}    train_acc = {train_acc}    avg_acc = {avg_acc}".format(
                        progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                        ce_loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                        avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                        train_acc=colored("{:.4f}".format(aggreeings.float().mean().cpu().numpy()), "blue",
                                          attrs=['bold']),
                        avg_acc=colored("{:.4f}".format(train_accuracy), "red", attrs=['bold'])))
                sys.stdout.flush()
        sys.stdout.write("\n")
        # if cfg.dataset.question_type == 'count':
        #     if (epoch + 1) % 5 == 0:
        #         optimizer = step_decay(cfg, optimizer)
        # else:
        #     if (epoch + 1) % 10 == 0:
        #         optimizer = step_decay(cfg, optimizer)
        sys.stdout.flush()
        logging.info("Epoch = %s   avg_loss = %.3f    avg_acc = %.3f" % (epoch, avg_loss, train_accuracy))

        if cfg.val.flag:
            output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                assert os.path.isdir(output_dir)
            valid_acc, *valid_output= validate(cfg, model, val_loader, device, write_preds=True)
            scheduler.step(valid_acc)
            if (valid_acc > best_val):
                best_val = valid_acc
                best_what = valid_output[0]
                best_who = valid_output[1]
                best_when = valid_output[3]
                best_how = valid_output[2]
                best_where = valid_output[4]
                    # Save best model
                ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                else:
                    assert os.path.isdir(ckpt_dir)

                save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(ckpt_dir, lctime+'_model.pt'))
                sys.stdout.write('\n >>>>>> save to %s <<<<<< \n' % (ckpt_dir))
                sys.stdout.flush()

            logging.info('~~~~~~ Valid Accuracy: %.4f ~~~~~~~' % valid_acc)
            logging.info('~~~~~~ Valid What Accuracy: %.4f ~~~~~~~' % valid_output[4])
            logging.info('~~~~~~ Valid Who Accuracy: %.4f ~~~~~~' % valid_output[5])
            logging.info('~~~~~~ Valid How Accuracy: %.4f ~~~~~~' % valid_output[6])
            logging.info('~~~~~~ Valid When Accuracy: %.4f ~~~~~~' % valid_output[7])
            logging.info('~~~~~~ Valid Where Accuracy: %.4f ~~~~~~' % valid_output[8])

            sys.stdout.write('~~~~~~ Valid Accuracy: {valid_acc}, What Accuracy: {what_acc}, Who Accuracy: {who_acc}, How Accuracy: {how_acc}, When Accuracy: {when_acc}, Where Accuracy: {where_acc} ~~~~~~~\n'.format(
                    valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold']),
                    what_acc=colored("{:.4f}".format(valid_output[4]), "red", attrs=['bold']),
                    who_acc=colored('{:.4f}'.format(valid_output[5]), "red", attrs=['bold']),
                    how_acc=colored('{:.4f}'.format(valid_output[6]), "red", attrs=['bold']),
                    when_acc=colored('{:.4f}'.format(valid_output[7]), "red", attrs=['bold']),
                    where_acc=colored('{:.4f}'.format(valid_output[8]), "red", attrs=['bold'])
                    ))
            sys.stdout.flush()

# Credit https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


def step_decay(cfg, optimizer):
    # compute the new learning rate based on decay rate
    cfg.train.lr *= 0.5
    logging.info("Reduced learning rate to {}".format(cfg.train.lr))
    sys.stdout.flush()
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.train.lr

    return optimizer


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    predicted = predicted.detach().argmax(1)
    agreeing = (predicted == true)
    return agreeing


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
    }
    time.sleep(10)
    torch.save(state, filename)


def main():
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/msvd_qa_swin.yml', type=str)
    args = parser.parse_args()
    cfg.dataset.name='tgif-qa'
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert cfg.dataset.name in ['sutd-qa','tgif-qa', 'msrvtt-qa', 'msvd-qa']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)
    # check if k_max is set correctly
    # assert cfg.train.k_max_frame_level <= 16
    # assert cfg.train.k_max_clip_level <= 8


    if not cfg.multi_gpus:
        torch.cuda.set_device(cfg.gpu_id)
    # make logging.info display into both shell and file
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    if not os.path.exists(cfg.dataset.save_dir):
        os.makedirs(cfg.dataset.save_dir)
    else:
        assert os.path.isdir(cfg.dataset.save_dir)
    log_file = os.path.join(cfg.dataset.save_dir, "log")
    if not cfg.train.restore and not os.path.exists(log_file):
        os.mkdir(log_file)
    else:
        assert os.path.isdir(log_file)

    fileHandler = logging.FileHandler(os.path.join(log_file, 'stdout.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(cfg).items():
        logging.info(k + ':' + str(v))
    # concat absolute path of input files

    if cfg.dataset.name == 'tgif-qa':
        cfg.dataset.vocab_json = '{}_{}_vocab_keybert.json'
        cfg.dataset.vocab_subject_json = '{}_{}_vocab_subject_keybert.json'
        cfg.dataset.vocab_relation_json = '{}_{}_vocab_relation_keybert.json'
        cfg.dataset.vocab_object_json = '{}_{}_vocab_object_keybert.json'        
        cfg.dataset.train_question_pt = '{}_{}_train_questions_keybert.pt'
        cfg.dataset.train_question_subject_pt = '{}_{}_train_questions_subject_keybert.pt'
        cfg.dataset.train_question_relation_pt = '{}_{}_train_questions_relation_keybert.pt'
        cfg.dataset.train_question_object_pt = '{}_{}_train_questions_object_keybert.pt'
        cfg.dataset.val_question_pt = '{}_{}_test_questions_keybert.pt'
        cfg.dataset.val_question_subject_pt = '{}_{}_test_questions_subject_keybert.pt'
        cfg.dataset.val_question_relation_pt = '{}_{}_test_questions_relation_keybert.pt'
        cfg.dataset.val_question_object_pt = '{}_{}_test_questions_object_keybert.pt'

        cfg.dataset.train_question_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.train_question_subject_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_subject_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.train_question_relation_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_relation_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.train_question_object_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_object_pt.format(cfg.dataset.name, cfg.dataset.question_type))                                            
        
        cfg.dataset.val_question_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.val_question_subject_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_subject_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.val_question_relation_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_relation_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.val_question_object_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_object_pt.format(cfg.dataset.name, cfg.dataset.question_type))                                               
        
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.vocab_subject_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_subject_json.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.vocab_relation_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_relation_json.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.vocab_object_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_object_json.format(cfg.dataset.name, cfg.dataset.question_type))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.appearance_dict = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_dict.format(cfg.dataset.name))
        cfg.dataset.motion_dict = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_dict.format(cfg.dataset.name))
    else:
        cfg.dataset.question_type = 'none'
        # cfg.dataset.appearance_feat = '{}_appearance_feat_swin_large.h5'
        # cfg.dataset.motion_feat = '{}_motion_feat_swin_large.h5'
        # cfg.dataset.appearance_dict = '{}_appearance_feat_swin_large_dict.h5'
        # cfg.dataset.motion_dict = '{}_motion_feat_swin_large_dict.h5'
        cfg.dataset.vocab_json = '{}_vocab_keybert_v3.json'
        cfg.dataset.vocab_subject_json = '{}_vocab_subject_keybert_v3.json'
        cfg.dataset.vocab_relation_json = '{}_vocab_relation_keybert_v3.json'
        cfg.dataset.vocab_object_json = '{}_vocab_object_keybert_v3.json'        
        cfg.dataset.train_question_pt = '{}_train_question_keybert_v3.pt'
        cfg.dataset.train_question_subject_pt = '{}_train_questions_subject_keybert_v3.pt'
        cfg.dataset.train_question_relation_pt = '{}_train_questions_relation_keybert_v3.pt'
        cfg.dataset.train_question_object_pt = '{}_train_questions_object_keybert_v3.pt'
        cfg.dataset.val_question_pt = '{}_test_question_keybert_v3.pt'
        cfg.dataset.val_question_subject_pt = '{}_test_questions_subject_keybert_v3.pt'
        cfg.dataset.val_question_relation_pt = '{}_test_questions_relation_keybert_v3.pt'
        cfg.dataset.val_question_object_pt = '{}_test_questions_object_keybert_v3.pt'
        cfg.dataset.train_question_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_pt.format(cfg.dataset.name))
        cfg.dataset.train_question_subject_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_subject_pt.format(cfg.dataset.name))
        cfg.dataset.train_question_relation_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_relation_pt.format(cfg.dataset.name))
        cfg.dataset.train_question_object_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_object_pt.format(cfg.dataset.name))

        cfg.dataset.val_question_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_pt.format(cfg.dataset.name))
        cfg.dataset.val_question_subject_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_subject_pt.format(cfg.dataset.name))
        cfg.dataset.val_question_relation_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_relation_pt.format(cfg.dataset.name))
        cfg.dataset.val_question_object_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_object_pt.format(cfg.dataset.name))

        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))
        cfg.dataset.vocab_subject_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_subject_json.format(cfg.dataset.name))
        cfg.dataset.vocab_relation_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_relation_json.format(cfg.dataset.name))
        cfg.dataset.vocab_object_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_object_json.format(cfg.dataset.name))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))
        cfg.dataset.appearance_dict = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_dict.format(cfg.dataset.name))
        cfg.dataset.motion_dict = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_dict.format(cfg.dataset.name))

    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train_oie(cfg)


if __name__ == '__main__':
    main()
