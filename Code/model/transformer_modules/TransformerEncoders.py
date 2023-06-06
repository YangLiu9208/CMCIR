import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from .rpe_2d import build_rpe, get_rpe_config
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionalEncodingLearned1D(nn.Module):
    """
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncodingLearned1D(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingLearned1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pos_embed = nn.Embedding(max_len, d_model)
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        idx = torch.arange(x.shape[0],device=x.device)
        x = x + self.pos_embed(idx).unsqueeze(1)
        return self.dropout(x)

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=512, s=1,dropout=0.1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 1, s, 0, bias=True, groups=embed_dim), )
        #self.proj = nn.Sequential(nn.Conv3d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s
        #self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        N, B, C = x.shape
        feat_token = x
        cnn_feat = feat_token.permute(1,2,0).contiguous().reshape(B,C,N,1)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).permute(2,0,1).contiguous() 
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class RPEAttention(nn.Module):
    '''
    Attention with 2D relative position encoding
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # 2D relative position encoding
        #config = get_rpe_config(skip=1)
        self.rpe_q, self.rpe_k, self.rpe_v = \
            build_rpe(None,
                      head_dim=head_dim,
                      num_heads=num_heads)

    def forward(self, x):
        N, B, C = x.shape
        x = x.permute(1,0,2)
        #B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q *= self.scale

        attn = (q @ k.transpose(-2, -1))

        # 2D relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q)

        # 2D relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # 2D relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(1,0,2)
        return x

class TransformerEncoderLayer_QKV(nn.Module):
    """
    Completed TransformerEncoderLayer for input Q K V

    """
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, res_dropout=0.1, activ_dropout=0.1, activation='gelu'):
        super(TransformerEncoderLayer_QKV,self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,dropout=attn_dropout)
        self.res_dropout = res_dropout
        self.activ_dropout = activ_dropout
        self.normalize_before = True

        self.fc1 = nn.Linear(embed_dim, 4*embed_dim)
        self.fc2 = nn.Linear(4*embed_dim, embed_dim)
        if activation=='relu':
            self.activ=nn.ReLU()
        if activation=='prelu':
            self.activ=nn.PReLU()
        if activation=='elu':
            self.activ=nn.ELU()
        if activation=='gelu':
            self.activ=nn.GELU()
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_key_padding_mask(self, max_length, key_len):
        #return shape (batch, max_length)
        return torch.arange(0,max_length,device=key_len.device).unsqueeze(0).expand(key_len.shape[0],max_length).ge(key_len.unsqueeze(1)).bool()

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x
  
    def forward(self, x_q, x_k=None, key_len=None):
        """
        Args:
            x_q (Tensor): input to the layer of shape (seq_len, batch, embed_dim)
            x_k (None or Tensor): if tensor, input to the layer of shape (seq_len', batch, embed_dim)
            key_len (None or Tensor): if Tensor, input to the layer of shape (batch,)
        Returns:
            out shape (seq_len, batch, embed_dim)
        """
        residual = x_q
        x_q = self.maybe_layer_norm(0, x_q, before=True)
        if x_k is None:
            key_padding_mask = self._get_key_padding_mask(x_q.shape[0],key_len) if key_len is not None else None
            x_q, _ = self.self_attn(query=x_q, key=x_q, value=x_q, key_padding_mask=key_padding_mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True) 
            key_padding_mask = self._get_key_padding_mask(x_k.shape[0],key_len) if key_len is not None else None
            x_q, _ = self.self_attn(query=x_q, key=x_k, value=x_k, key_padding_mask=key_padding_mask)
        x_q = F.dropout(x_q, p=self.res_dropout, training=self.training)
        x_q = residual + x_q
        x_q = self.maybe_layer_norm(0, x_q, after=True)

        residual = x_q
        x_q = self.maybe_layer_norm(1, x_q, before=True)
        x_q = self.activ(self.fc1(x_q))
        x_q = F.dropout(x_q, p=self.activ_dropout, training=self.training)
        x_q = self.fc2(x_q)
        x_q = F.dropout(x_q, p=self.res_dropout, training=self.training)
        x_q = residual + x_q
        x_q = self.maybe_layer_norm(1, x_q, after=True)
        return x_q






class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, pos_flag='learned',pos_dropout=0.1, num_heads=4, attn_dropout=0.1, res_dropout=0.1, activ_dropout=0.1, activation='gelu', num_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_flag = pos_flag
        if pos_flag=='sincos':
            self.embed_scale = math.sqrt(embed_dim)
            self.pos_encoder = PositionalEncoding(embed_dim, pos_dropout)
        if pos_flag=='learned':
            self.embed_scale = 1.0
            self.pos_encoder = PositionalEncodingLearned1D(embed_dim, pos_dropout)
        if pos_flag=='poscnn':
            self.embed_scale = 1.0
            self.pos_encoder = PosCNN(embed_dim,embed_dim, 1,pos_dropout)
        if pos_flag=='relative':
            self.embed_scale = 1.0
            self.pos_encoder = RPEAttention(dim=embed_dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)

        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            new_layer = TransformerEncoderLayer_QKV(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                res_dropout=res_dropout,
                                                activ_dropout=activ_dropout,
                                                activation=activation)
            self.layers.append(new_layer)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_q, x_k=None, key_len=None):
        """
        Args:
            x_q (Tensor): input to the layer of shape (seq_len, batch, embed_dim)
            x_k (None or Tensor): if tensor, input to the layer of shape (seq_len', batch, embed_dim)
            key_len (None or Tensor): if Tensor, input to the layer of shape (batch,)
        Returns:
            out shape (seq_len, batch, embed_dim)
        """
        x_q = self.pos_encoder(self.embed_scale*x_q)
        if x_k is not None:
            x_k = self.pos_encoder(self.embed_scale*x_k)
         # encoder layers
        intermediates = [x_q]
        for layer in self.layers:
            x_q = layer(x_q, x_k, key_len)
            intermediates.append(x_q)

        x_q = self.layer_norm(x_q)
        return x_q




class TransformerEncoder_nopos(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, res_dropout=0.1, activ_dropout=0.1, activation='gelu', num_layers=6):
        super().__init__()

        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            new_layer = TransformerEncoderLayer_QKV(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                res_dropout=res_dropout,
                                                activ_dropout=activ_dropout,
                                                activation=activation)
            self.layers.append(new_layer)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_q, x_k=None, key_len=None):
        """
        Args:
            x_q (Tensor): input to the layer of shape (seq_len, batch, embed_dim)
            x_k (None or Tensor): if tensor, input to the layer of shape (seq_len', batch, embed_dim)
            key_len (None or Tensor): if Tensor, input to the layer of shape (batch,)
        Returns:
            out shape (seq_len, batch, embed_dim)
        """
         # encoder layers
        intermediates = [x_q]
        for layer in self.layers:
            x_q = layer(x_q, x_k, key_len)
            intermediates.append(x_q)

        x_q = self.layer_norm(x_q)
        return x_q