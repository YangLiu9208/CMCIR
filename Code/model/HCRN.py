import numpy as np
from torch.nn import functional as F

from .utils import *
from .CRN import CRN
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from model.transformer_modules.TransformerEncoders import *
#from .vivit import ViViT, ViT
#from .build import build_model, build_model3d
class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill

class FeatureAggregation_four(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation_four, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(2*module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(2*module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, question_rep_subject, question_rep_relation, question_object, visual_feat, visual_feat_subject, visual_feat_relation, visual_feat_object):
        visual_feat = self.dropout(visual_feat)
        visual_feat_subject = self.dropout(visual_feat_subject)
        visual_feat_relation = self.dropout(visual_feat_relation)
        visual_feat_object = self.dropout(visual_feat_object)
        q_proj = self.q_proj(question_rep)
        q_proj_subject = self.q_proj(question_rep_subject)
        q_proj_relation = self.q_proj(question_rep_relation)
        q_proj_object = self.q_proj(question_object)
        v_proj = self.v_proj(visual_feat)
        v_proj_subject = self.v_proj(visual_feat_subject)
        v_proj_relation = self.v_proj(visual_feat_relation)
        v_proj_object = self.v_proj(visual_feat_object)


        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat_subject = torch.cat((v_proj_subject, q_proj_subject.unsqueeze(1) * v_proj_subject), dim=-1)
        v_q_cat_relation = torch.cat((v_proj_relation, q_proj_relation.unsqueeze(1) * v_proj_relation), dim=-1)
        v_q_cat_object = torch.cat((v_proj_object, q_proj_object.unsqueeze(1) * v_proj_object), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat_subject = self.cat(v_q_cat_subject)
        v_q_cat_relation = self.cat(v_q_cat_relation)
        v_q_cat_object = self.cat(v_q_cat_object)

        v_q_cat = self.activation(v_q_cat)
        v_q_cat_subject = self.activation(v_q_cat_subject)
        v_q_cat_relation = self.activation(v_q_cat_relation)
        v_q_cat_object = self.activation(v_q_cat_object)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn_subject = self.attn(v_q_cat_subject)  # (bz, k, 1)
        attn_relation = self.attn(v_q_cat_relation)  # (bz, k, 1)
        attn_object = self.attn(v_q_cat_object)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)
        attn_subject = F.softmax(attn_subject, dim=1)  # (bz, k, 1)
        attn_relation = F.softmax(attn_relation, dim=1)  # (bz, k, 1)
        attn_object = F.softmax(attn_object, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)
        v_distill_subject = (attn_subject * visual_feat_subject).sum(1)
        v_distill_relation = (attn_relation * visual_feat_relation).sum(1)
        v_distill_object = (attn_object * visual_feat_object).sum(1)
        #v_distill = (attn *visual_feat).sum(1)

        return torch.cat((v_distill,v_distill_subject, v_distill_relation, v_distill_object ),1)

class InputUnitLinguistic(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        #self.encoder = nn.GRU(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len.cpu(), batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        #_, question_embedding= self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding

class InputUnitLinguistic_subject(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic_subject, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        #self.encoder = nn.GRU(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len.cpu(), batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        #_, question_embedding= self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding

class InputUnitLinguistic_relation(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic_relation, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        #self.encoder = nn.GRU(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len.cpu(), batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        #_, question_embedding= self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding

class InputUnitLinguistic_object(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic_object, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        #self.encoder = nn.GRU(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len.cpu(), batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        #_, question_embedding= self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding

class InputUnitLinguistic_Transformer(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512):
        super(InputUnitLinguistic_Transformer, self).__init__()

        self.activ=nn.GELU()
        self.dim = module_dim

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, rnn_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=rnn_dim, pos_flag='learned',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=6)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        return questions_embedding

class InputUnitLinguistic_subject_Transformer(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512):
        super(InputUnitLinguistic_subject_Transformer, self).__init__()

        self.activ=nn.GELU()
        self.dim = module_dim

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, rnn_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=rnn_dim, pos_flag='learned',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=6)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        return questions_embedding

class InputUnitLinguistic_relation_Transformer(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512):
        super(InputUnitLinguistic_relation_Transformer, self).__init__()

        self.activ=nn.GELU()
        self.dim = module_dim

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, rnn_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=rnn_dim, pos_flag='learned',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=6)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        return questions_embedding

class InputUnitLinguistic_object_Transformer(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512):
        super(InputUnitLinguistic_object_Transformer, self).__init__()

        self.activ=nn.GELU()
        self.dim = module_dim

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, rnn_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=rnn_dim, pos_flag='learned',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=6)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        return questions_embedding

class InputUnitVisual(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512):
        super(InputUnitVisual, self).__init__()

        self.clip_level_motion_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.clip_level_question_cond = CRN(module_dim, k_max_frame_level-2, k_max_frame_level-2, gating=True, spl_resolution=spl_resolution)
        self.video_level_motion_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level-2, k_max_clip_level-2, gating=True, spl_resolution=spl_resolution)

        self.sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        self.appearance_feat_proj = nn.Linear(4*module_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)


        #self.hid = 8
        #self.in_head = 8
        #self.out_head = 1

        #self.appearance_gcn1 = GATConv(module_dim, module_dim, heads=self.in_head, dropout=0.1)
        #self.appearance_gcn2 = GATConv(self.in_head*module_dim, module_dim, concat=False, heads=self.out_head, dropout=0.1)
        #self.motion_gcn1 = GATConv(module_dim, module_dim, heads=self.in_head, dropout=0.1)
        #self.motion_gcn2 = GATConv(self.in_head*module_dim, module_dim, concat=False, heads=self.out_head, dropout=0.1)

        self.appearance_gcn1=GCNConv(module_dim, module_dim)
        self.appearance_gcn2=GCNConv(module_dim, module_dim)
        self.motion_gcn1=GCNConv(module_dim, module_dim)
        self.motion_gcn2=GCNConv(module_dim, module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        #edge_index_appearance=torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15],\
        #    [1,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9,8,10,9,11,10,12,11,13,12,14,13,15,14]],dtype=torch.long).cuda()
        edge_index_appearance=torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15],\
            [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]],dtype=torch.long).cuda()
        #edge_index_motion=torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6,6,7],\
        #    [1,0,2,1,3,2,4,3,5,4,6,5,7,6]],dtype=torch.long).cuda()
        edge_index_motion=torch.tensor([[0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7],\
            [1,2,3,4,5,6,7,0,2,3,4,5,6,7,0,1,3,4,5,6,7,0,1,2,4,5,6,7,0,1,2,3,5,6,7,0,1,2,3,4,6,7,0,1,2,3,4,5,7,0,1,2,3,4,5,6]],dtype=torch.long).cuda()
        motion_video_feat_gcn=[]
        appearance_video_feat_gcn=[]
        clip_level_motion_projs=[]
        clip_level_appearance_projs=[]
        for i in range(appearance_video_feat.size(1)):     
            clip_level_motion = motion_video_feat[:, i, :]  # (bz, 2048)
            clip_level_motion_projs.append(self.clip_level_motion_proj(clip_level_motion))
            clip_level_appearance = appearance_video_feat[:, i, :, :]  # (bz, 16, 2048)
            clip_level_appearance_projs.append(self.appearance_feat_proj(clip_level_appearance))  # (bz, 16, 512)
        clip_level_motion_proj_stack=torch.stack(clip_level_motion_projs).permute([1,0,2])
        clip_level_appearance_proj_stack=torch.stack(clip_level_appearance_projs).permute([1,0,2,3])

        for j in range(motion_video_feat.size(0)):
            motion_video_feat_gcn.append(self.motion_gcn2(self.motion_gcn1(clip_level_motion_proj_stack[j,:,:],edge_index_motion),edge_index_motion))
            appearance_video_feat_gcn.append(self.appearance_gcn2(self.appearance_gcn1(clip_level_appearance_proj_stack[j,:,:,:],edge_index_appearance),edge_index_appearance))
        clip_level_motion_proj_gcn=torch.stack(motion_video_feat_gcn)
        clip_level_appearance_proj_gcn=torch.stack(appearance_video_feat_gcn)

        for k in range(appearance_video_feat.size(1)):
            # clip level CRNs
            clip_level_crn_motion = self.clip_level_motion_cond(torch.unbind(clip_level_appearance_proj_gcn[:, k, :, :], dim=1),
                                                                clip_level_motion_proj_gcn[:, k, :])
            clip_level_crn_question = self.clip_level_question_cond(clip_level_crn_motion, question_embedding_proj)

            clip_level_crn_output = torch.cat(
                [frame_relation.unsqueeze(1) for frame_relation in clip_level_crn_question],
                dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim)
            clip_level_crn_outputs.append(clip_level_crn_output)

        # Encode video level motion
        _, (video_level_motion, _) = self.sequence_encoder(motion_video_feat)
        #_, video_level_motion = self.sequence_encoder(motion_video_feat)
        video_level_motion = video_level_motion.transpose(0, 1)
        video_level_motion_feat_proj = self.video_level_motion_proj(video_level_motion)
        # video level CRNs
        video_level_crn_motion = self.video_level_motion_cond(clip_level_crn_outputs, video_level_motion_feat_proj)
        video_level_crn_question = self.video_level_question_cond(video_level_crn_motion,
                                                                  question_embedding_proj.unsqueeze(1))

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_question],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)

        return video_level_crn_output

class InputUnitVisual_Transformer(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, module_dim=512):
        super(InputUnitVisual_Transformer, self).__init__()

        self.appearance_transformer= build_model()
        self.motion_transformer = build_model3d()
        # self.appearance_transformer= ViT()
        # self.motion_transformer = ViViT(224, 16, 100, 16)

        self.clip_level_motion_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.clip_level_question_cond = CRN(module_dim, k_max_frame_level-2, k_max_frame_level-2, gating=True, spl_resolution=spl_resolution)
        self.video_level_motion_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level-2, k_max_clip_level-2, gating=True, spl_resolution=spl_resolution)

        self.sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)


        #self.hid = 8
        #self.in_head = 8
        #self.out_head = 1

        #self.appearance_gcn1 = GATConv(module_dim, module_dim, heads=self.in_head, dropout=0.1)
        #self.appearance_gcn2 = GATConv(self.in_head*module_dim, module_dim, concat=False, heads=self.out_head, dropout=0.1)
        #self.motion_gcn1 = GATConv(module_dim, module_dim, heads=self.in_head, dropout=0.1)
        #self.motion_gcn2 = GATConv(self.in_head*module_dim, module_dim, concat=False, heads=self.out_head, dropout=0.1)

        self.appearance_gcn1=GCNConv(module_dim, module_dim)
        self.appearance_gcn2=GCNConv(module_dim, module_dim)
        self.motion_gcn1=GCNConv(module_dim, module_dim)
        self.motion_gcn2=GCNConv(module_dim, module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video, motion_video, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        self.sequence_encoder.flatten_parameters()
        appearance_video_feat = []
        motion_video_feat = []
        for m in range(motion_video.size(0)): 
            motion_video_feat.append(self.motion_transformer(motion_video[m,:,:,:,:,:]))
            for n in range(motion_video.size(1)): 
                appearance_video_feat.append(self.appearance_transformer(appearance_video[m,n,:,:,:,:]))
        appearance_video_feat=torch.stack(appearance_video_feat).view(appearance_video.size(0),appearance_video.size(1),appearance_video.size(2),-1)
        motion_video_feat = torch.stack(motion_video_feat)
        del motion_video
        del appearance_video
        # for m in range(motion_video.size(1)): 
        #     motion_video_feat.append(self.motion_transformer(motion_video[:,m,:,:,:,:]))
        #     appearance_video_feat.append(self.appearance_transformer(motion_video[:,m,:,:,:,:].reshape(motion_video.size(0)*motion_video.size(2), motion_video.size(3), motion_video.size(4),motion_video.size(5))))
        # appearance_video_feat=torch.stack(appearance_video_feat).permute(1,0,2).view(motion_video.size(0),motion_video.size(2),motion_video.size(1),-1).permute(0,2,1,3)
        # motion_video_feat = torch.stack(motion_video_feat).permute(1,0,2)


        batch_size = motion_video_feat.size(0)
        clip_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        #edge_index_appearance=torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15],\
        #    [1,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9,8,10,9,11,10,12,11,13,12,14,13,15,14]],dtype=torch.long).cuda()
        edge_index_appearance=torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15],\
             [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]],dtype=torch.long).cuda()
        # edge_index_motion=torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6,6,7],\
        #     [1,0,2,1,3,2,4,3,5,4,6,5,7,6]],dtype=torch.long).cuda()
        edge_index_motion=torch.tensor([[0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7],\
            [1,2,3,4,5,6,7,0,2,3,4,5,6,7,0,1,3,4,5,6,7,0,1,2,4,5,6,7,0,1,2,3,5,6,7,0,1,2,3,4,6,7,0,1,2,3,4,5,7,0,1,2,3,4,5,6]],dtype=torch.long).cuda()
        motion_video_feat_gcn=[]
        appearance_video_feat_gcn=[]
        clip_level_motion_projs=[]
        clip_level_appearance_projs=[]
        for i in range(motion_video_feat.size(1)):     
            clip_level_motion = motion_video_feat[:, i, :]  # (bz, 2048)
            clip_level_motion_projs.append(self.clip_level_motion_proj(clip_level_motion))
            clip_level_appearance = appearance_video_feat[:, i, :, :]  # (bz, 16, 2048)
            clip_level_appearance_projs.append(self.appearance_feat_proj(clip_level_appearance))  # (bz, 16, 512)

        clip_level_motion_projs=torch.stack(clip_level_motion_projs).permute([1,0,2])
        clip_level_appearance_projs=torch.stack(clip_level_appearance_projs).permute([1,0,2,3])

        for j in range(motion_video_feat.size(0)):
            motion_video_feat_gcn.append(self.motion_gcn2(self.motion_gcn1(clip_level_motion_projs[j,:,:],edge_index_motion),edge_index_motion))
            appearance_video_feat_gcn.append(self.appearance_gcn2(self.appearance_gcn1(clip_level_appearance_projs[j,:,:,:],edge_index_appearance),edge_index_appearance))
        clip_level_motion_proj_gcn=torch.stack(motion_video_feat_gcn)
        clip_level_appearance_proj_gcn=torch.stack(appearance_video_feat_gcn)

        for k in range(appearance_video_feat.size(1)):
            # clip level CRNs
            clip_level_crn_motion = self.clip_level_motion_cond(torch.unbind(clip_level_appearance_proj_gcn[:, k, :, :], dim=1),
                                                                clip_level_motion_proj_gcn[:, k, :])
            clip_level_crn_question = self.clip_level_question_cond(clip_level_crn_motion, question_embedding_proj)

            clip_level_crn_output = torch.cat(
                [frame_relation.unsqueeze(1) for frame_relation in clip_level_crn_question],
                dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim)
            clip_level_crn_outputs.append(clip_level_crn_output)

        # Encode video level motion
        _, (video_level_motion, _) = self.sequence_encoder(motion_video_feat)
        # del appearance_video_feat
        # del motion_video_feat
        #_, video_level_motion = self.sequence_encoder(motion_video_feat)
        video_level_motion = video_level_motion.transpose(0, 1)
        video_level_motion_feat_proj = self.video_level_motion_proj(video_level_motion)
        # video level CRNs
        video_level_crn_motion = self.video_level_motion_cond(clip_level_crn_outputs, video_level_motion_feat_proj)
        video_level_crn_question = self.video_level_question_cond(video_level_crn_motion,
                                                                  question_embedding_proj.unsqueeze(1))

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_question],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim)

        return video_level_crn_output


class InputUnitVisual_GTransformer(nn.Module):
    def __init__(self, motion_dim, appearance_dim, module_dim=512):
        super(InputUnitVisual_GTransformer, self).__init__()

        self.clip_level_motion_proj = nn.Sequential(
                                                nn.Linear(motion_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )
        self.appearance_feat_proj = nn.Sequential(
                                                nn.Linear(appearance_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )

        self.QMTransformer =  TransformerEncoder(embed_dim=512, pos_flag='learned',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.QATransformer =  TransformerEncoder(embed_dim=512, pos_flag='learned',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIMTransformer =  TransformerEncoder(embed_dim=512, pos_flag='learned',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIATransformer =  TransformerEncoder(embed_dim=512, pos_flag='learned',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.module_dim = module_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding, appearance_dict, motion_dict):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        visual_embedding_motion = self.clip_level_motion_proj(motion_video_feat).permute(1,0,2)
        visual_embedding_appearance   = self.appearance_feat_proj(appearance_video_feat).permute(1,2,0,3)

        #visual_embedding = torch.cat([visual_embedding_motion,visual_embedding_appearance],dim=1)
        #question_embedding_v  = self.QATransformer(question_embedding,torch.mean(visual_embedding,1))

        question_visual_m = self.QMTransformer(question_embedding,visual_embedding_motion)
        question_visual_a  = self.QATransformer(question_embedding,torch.mean(visual_embedding_appearance, 1))

        visual_embedding_m = self.VIMTransformer(visual_embedding_motion,question_visual_m)
        visual_embedding_a = self.VIATransformer(torch.mean(visual_embedding_appearance, 1),question_visual_a)

        #appearance_embedding_q = self.VIATransformer(torch.mean(visual_embedding_appearance,1),question_appearance_att)
        question_embedding_v = torch.cat([question_visual_m,question_visual_a], dim=2)
        visual_embedding = torch.cat([visual_embedding_m,visual_embedding_a], dim=2)
        

        return torch.mean(question_embedding_v,0), visual_embedding.permute(1,0,2)

class OutputUnitOpenEnded(nn.Module):
    def __init__(self, module_dim=512, num_answers=1000):
        super(OutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(8*module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 9, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitMultiChoices(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoices, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 4, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding, ans_candidates_embedding,
                a_visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        out = torch.cat([q_visual_embedding, question_embedding, a_visual_embedding,
                         ans_candidates_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitCount(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitCount, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.regression = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.regression(out)

        return out


class SE_Fusion_Four(nn.Module):
    def __init__(self,channel_1=128,channel_2=256,channel_3=256,channel_4=256,reduction=4):
        super(SE_Fusion_Four, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_3 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_3, bias=True),
            nn.Sigmoid()
        )
        self.fc_4 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_4, bias=True),
            nn.Sigmoid()
        )
        self.gcn=GCNConv(channel_1, channel_1)
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        self.gcn1 = GATConv(channel_1, channel_1, heads=self.in_head, dropout=0.1).cuda()
        self.gcn2 = GATConv(self.in_head*channel_1, channel_1, concat=False, heads=self.out_head, dropout=0.1).cuda()

    def forward(self, fea1, fea2, fea3, fea4):

        #squeeze
        c_1,D_1= fea1.size()
        y_1 = fea1

        c_2,D_2= fea2.size()
        y_2 = fea2

        c_3, D_3= fea3.size()
        y_3 = fea3

        c_4, D_4= fea4.size()
        y_4 = fea4

        y=torch.stack((y_1,y_2,y_3,y_4),1)
        y_gcn=[]
        y_gcn2=[]
        edge_index=torch.tensor([[0,1,0,2,0,3,1,2,1,3,2,3],[1,0,2,0,3,0,2,1,3,1,3,2]],dtype=torch.long).cuda()
        for j in range(y.size(0)):
            y_gcn.append(self.gcn(y[j,:,:],edge_index))
            # y_gcn.append(self.gcn1(y[j],edge_index))
            # y_gcn2.append(self.gcn2(y_gcn[j],edge_index))
        y_all_gcn=torch.stack(y_gcn)
        y_1=y_all_gcn[:,0,:]
        y_2=y_all_gcn[:,1,:]
        y_3=y_all_gcn[:,2,:]
        y_4=y_all_gcn[:,3,:]

        z=torch.cat((y_1,y_2,y_3,y_4),1)

        y_1 =self.fc_1(z).view(c_1, D_1)
        y_2 = self.fc_2(z).view(c_2, D_2)  
        y_3 = self.fc_3(z).view(c_3, D_3) 
        y_4 = self.fc_4(z).view(c_4, D_4) 
        
        return torch.mul((F.relu(y_1)).expand_as(fea1),fea1), torch.mul((F.relu(y_2)).expand_as(fea2),fea2), torch.mul((F.relu(y_3)).expand_as(fea3),fea3), torch.mul((F.relu(y_4)).expand_as(fea4),fea4)
        #return y_1, y_2, y_3, y_4
                
class HCRNNetwork(nn.Module):
    def __init__(self, vision_dim, module_dim, word_dim, k_max_frame_level, k_max_clip_level, spl_resolution, vocab, vocab_subject, vocab_relation, vocab_object, question_type):
        super(HCRNNetwork, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)
        self.feature_aggregation_four = FeatureAggregation_four(module_dim)

        if self.question_type in ['action', 'transition']:
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        elif self.question_type == 'count':
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitCount(module_dim=module_dim)
        else:
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            encoder_vocab_subject_size = len(vocab_subject['question_token_to_idx'])
            encoder_vocab_relation_size = len(vocab_relation['question_token_to_idx'])
            encoder_vocab_object_size = len(vocab_object['question_token_to_idx'])
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.linguistic_subject_input_unit = InputUnitLinguistic_subject(vocab_size=encoder_vocab_subject_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.linguistic_relation_input_unit = InputUnitLinguistic_relation(vocab_size=encoder_vocab_relation_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.linguistic_object_input_unit = InputUnitLinguistic_object(vocab_size=encoder_vocab_object_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)                                                                                                         
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitOpenEnded(num_answers=self.num_classes)
            self.fusion=SE_Fusion_Four(512,512,512,512,8)

        init_modules(self.modules(), w_init="xavier_uniform")
        nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, question,question_subject, question_relation, question_object, question_len, question_subject_len, question_relation_len, question_object_len):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)
        if self.question_type in ['frameqa', 'count', 'none']:
            # get image, word, and sentence embeddings
            question_embedding = self.linguistic_input_unit(question, question_len)
            question_embedding_subject = self.linguistic_subject_input_unit(question_subject, question_subject_len)
            question_embedding_relation = self.linguistic_relation_input_unit(question_relation, question_relation_len)
            question_embedding_object = self.linguistic_object_input_unit(question_object, question_object_len) 
            question_embedding,question_embedding_subject,question_embedding_relation, question_embedding_object=self.fusion(question_embedding,question_embedding_subject,question_embedding_relation,question_embedding_object)
            #question_embedding = torch.cat((question_embedding, question_embedding_subject, question_embedding_relation, question_embedding_object), 1)
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)
            visual_embedding_subject = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_subject)
            visual_embedding_relation = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_relation)
            visual_embedding_object = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_object)

            #visual_embedding = self.feature_aggregation(question_embedding, visual_embedding_subject)
            visual_embedding = self.feature_aggregation_four(question_embedding, visual_embedding, visual_embedding_subject, visual_embedding_relation, visual_embedding_object)

            out = self.output_unit(question_embedding, visual_embedding)
        else:
            question_embedding = self.linguistic_input_unit(question, question_len)
                       
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            # ans_candidates: (batch_size, num_choices, max_len)
            ans_candidates_agg = ans_candidates.view(-1, ans_candidates.size(2))
            ans_candidates_len_agg = ans_candidates_len.view(-1)

            batch_agg = np.reshape(
                np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])

            ans_candidates_embedding = self.linguistic_input_unit(ans_candidates_agg, ans_candidates_len_agg)

            a_visual_embedding = self.feature_aggregation(ans_candidates_embedding, visual_embedding[batch_agg])
            out = self.output_unit(question_embedding[batch_agg], q_visual_embedding[batch_agg],
                                   ans_candidates_embedding,
                                   a_visual_embedding)
        return out

class HCRTransformer(nn.Module):
    def __init__(self, vision_dim,  module_dim, word_dim, k_max_frame_level, k_max_clip_level, spl_resolution, vocab, vocab_subject, vocab_relation, vocab_object, question_type):
        super(HCRTransformer, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)
        self.feature_aggregation_four = FeatureAggregation_four(module_dim)

        if self.question_type in ['action', 'transition']:
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        elif self.question_type == 'count':
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitCount(module_dim=module_dim)
        else:
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            encoder_vocab_subject_size = len(vocab_subject['question_token_to_idx'])
            encoder_vocab_relation_size = len(vocab_relation['question_token_to_idx'])
            encoder_vocab_object_size = len(vocab_object['question_token_to_idx'])
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.linguistic_subject_input_unit = InputUnitLinguistic_subject(vocab_size=encoder_vocab_subject_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.linguistic_relation_input_unit = InputUnitLinguistic_relation(vocab_size=encoder_vocab_relation_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)
            self.linguistic_object_input_unit = InputUnitLinguistic_object(vocab_size=encoder_vocab_object_size, wordvec_dim=word_dim,
                                                             module_dim=module_dim, rnn_dim=module_dim)                                                                                                         
            self.visual_input_unit = InputUnitVisual_Transformer(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, module_dim=module_dim)
            self.output_unit = OutputUnitOpenEnded(num_answers=self.num_classes)
            self.fusion=SE_Fusion_Four(512,512,512,512,8)

        init_modules(self.modules(), w_init="xavier_uniform")
        nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, question,question_subject, question_relation, question_object, question_len, question_subject_len, question_relation_len, question_object_len):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)
        if self.question_type in ['frameqa', 'count', 'none']:
            # get image, word, and sentence embeddings
            question_embedding = self.linguistic_input_unit(question, question_len)
            #question_embedding_subject = self.linguistic_subject_input_unit(question_subject, question_subject_len)
            #question_embedding_relation = self.linguistic_relation_input_unit(question_relation, question_relation_len)
            #question_embedding_object = self.linguistic_object_input_unit(question_object, question_object_len) 
            #question_embedding,question_embedding_subject,question_embedding_relation, question_embedding_object=self.fusion(question_embedding,question_embedding_subject,question_embedding_relation,question_embedding_object)
            #question_embedding = torch.cat((question_embedding, question_embedding_subject, question_embedding_relation, question_embedding_object), 1)
            visual_embedding = self.visual_input_unit(video_appearance_feat,video_motion_feat, question_embedding)
            #visual_embedding_subject = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_subject)
            #visual_embedding_relation = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_relation)
            #visual_embedding_object = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_object)

            visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)
            #visual_embedding = self.feature_aggregation_four(question_embedding, visual_embedding, visual_embedding_subject, visual_embedding_relation, visual_embedding_object)

            out = self.output_unit(question_embedding, visual_embedding)
        else:
            question_embedding = self.linguistic_input_unit(question, question_len)
                       
            visual_embedding = self.visual_input_unit(video_motion_feat, question_embedding)

            q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            # ans_candidates: (batch_size, num_choices, max_len)
            ans_candidates_agg = ans_candidates.view(-1, ans_candidates.size(2))
            ans_candidates_len_agg = ans_candidates_len.view(-1)

            batch_agg = np.reshape(
                np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])

            ans_candidates_embedding = self.linguistic_input_unit(ans_candidates_agg, ans_candidates_len_agg)

            a_visual_embedding = self.feature_aggregation(ans_candidates_embedding, visual_embedding[batch_agg])
            out = self.output_unit(question_embedding[batch_agg], q_visual_embedding[batch_agg],
                                   ans_candidates_embedding,
                                   a_visual_embedding)
        return out


class STC_Transformer(nn.Module):
    def __init__(self, motion_dim, appearance_dim, module_dim, word_dim,  vocab, vocab_subject, vocab_relation, vocab_object, question_type):
        super(STC_Transformer, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)
        self.feature_aggregation_four = FeatureAggregation_four(module_dim)

        if self.question_type in ['action', 'transition']:
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.visual_input_unit = InputUnitVisual_GTransformer(motion_dim=motion_dim, appearance_dim=appearance_dim, module_dim=module_dim)
            self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        elif self.question_type == 'count':
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.visual_input_unit = InputUnitVisual_GTransformer(motion_dim=motion_dim, appearance_dim=appearance_dim, module_dim=module_dim)
            self.output_unit = OutputUnitCount(module_dim=module_dim)
        else:
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            encoder_vocab_subject_size = len(vocab_subject['question_token_to_idx'])
            encoder_vocab_relation_size = len(vocab_relation['question_token_to_idx'])
            encoder_vocab_object_size = len(vocab_object['question_token_to_idx'])
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic_Transformer(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_subject_input_unit = InputUnitLinguistic_subject_Transformer(vocab_size=encoder_vocab_subject_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_relation_input_unit = InputUnitLinguistic_relation_Transformer(vocab_size=encoder_vocab_relation_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_object_input_unit = InputUnitLinguistic_object_Transformer(vocab_size=encoder_vocab_object_size, wordvec_dim=word_dim,module_dim=module_dim)                                                                                                         
            self.visual_input_unit = InputUnitVisual_GTransformer(motion_dim=motion_dim, appearance_dim=appearance_dim, module_dim=module_dim)
            self.output_unit = OutputUnitOpenEnded(num_answers=self.num_classes)
            self.fusion=SE_Fusion_Four(1024,1024,1024,1024,8).cuda()

        init_modules(self.modules(), w_init="xavier_uniform")
        nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, question,question_subject, question_relation, question_object, question_len, question_subject_len, question_relation_len, question_object_len, appearance_dict, motion_dict):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)
        if self.question_type in ['frameqa', 'count', 'none']:
            # get image, word, and sentence embeddings
            question_embedding = self.linguistic_input_unit(question, question_len)
            question_embedding_subject = self.linguistic_subject_input_unit(question_subject, question_subject_len)
            question_embedding_relation = self.linguistic_relation_input_unit(question_relation, question_relation_len)
            question_embedding_object = self.linguistic_object_input_unit(question_object, question_object_len) 
            #question_embedding = torch.cat((question_embedding, question_embedding_subject, question_embedding_relation, question_embedding_object), 1)
            question_embedding_v, visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding, appearance_dict, motion_dict )
            question_embedding_subject_v, visual_embedding_subject = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_subject, appearance_dict, motion_dict )
            question_embedding_relation_v, visual_embedding_relation = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_relation, appearance_dict, motion_dict )
            question_embedding_object_v, visual_embedding_object = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_object, appearance_dict, motion_dict )


            question_embedding_v,question_embedding_subject_v,question_embedding_relation_v, question_embedding_object_v=self.fusion(question_embedding_v,question_embedding_subject_v,question_embedding_relation_v,question_embedding_object_v)
            question_embedding_all = torch.cat((question_embedding_v, question_embedding_subject_v, question_embedding_relation_v, question_embedding_object_v), 1)
            #visual_embedding = self.feature_aggregation(question_embedding_v, visual_embedding)
            visual_embedding_all = self.feature_aggregation_four(question_embedding_v, question_embedding_subject_v, question_embedding_relation_v, question_embedding_object_v, visual_embedding, visual_embedding_subject, visual_embedding_relation, visual_embedding_object)

            out = self.output_unit(question_embedding_all, visual_embedding_all)
        else:
            question_embedding = self.linguistic_input_unit(question, question_len)
                       
            visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding)

            q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

            # ans_candidates: (batch_size, num_choices, max_len)
            ans_candidates_agg = ans_candidates.view(-1, ans_candidates.size(2))
            ans_candidates_len_agg = ans_candidates_len.view(-1)

            batch_agg = np.reshape(
                np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 5]), [-1])

            ans_candidates_embedding = self.linguistic_input_unit(ans_candidates_agg, ans_candidates_len_agg)

            a_visual_embedding = self.feature_aggregation(ans_candidates_embedding, visual_embedding[batch_agg])
            out = self.output_unit(question_embedding[batch_agg], q_visual_embedding[batch_agg],
                                   ans_candidates_embedding,
                                   a_visual_embedding)
        return out