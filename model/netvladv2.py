import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import faiss
import numpy as np
import random

class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(2*module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(2*module_dim, module_dim, bias=False)

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

        v_distill = (attn * visual_feat)

        return v_distill

class GAFN(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False, use_faiss=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(GAFN, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 1.0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv1d(dim, num_clusters, kernel_size=1, bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.use_faiss = use_faiss
        self.bn = nn.BatchNorm2d(8)
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=2)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x.permute(0,2,1).contiguous())
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-2), -1, -1).permute(1, 0, 2).unsqueeze(0)
        residual *= soft_assign.unsqueeze(3)
        vlad = residual.permute(0, 2, 1, 3).contiguous()#.sum(dim=-2)

        vlad = F.normalize(vlad, p=2, dim=3)  # intra-normalization
        vlad = self.bn(vlad)
        vlad = vlad.reshape(x.size(0), x.size(1), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=2)  # L2 normalize

        return vlad

class GAFN_Four(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False, use_faiss=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(GAFN_Four, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 1.0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.use_faiss = use_faiss
        self.bn = nn.BatchNorm3d(8)
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C, D = x.shape[:3]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=3)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x.permute(0,3,1,2).contiguous())
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, D, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1, -1).permute(1, 0, 2, 3, 4).contiguous() - \
            self.centroids.expand(x_flatten.size(-3), -1, -1).expand(x_flatten.size(-2), -1, -1, -1).permute( 2, 1, 0, 3).contiguous().unsqueeze(0)
        residual *= soft_assign.unsqueeze(4)
        vlad = residual.permute(0, 2, 3, 1, 4).contiguous()#.sum(dim=-2)

        vlad = F.normalize(vlad, p=2, dim=4)  # intra-normalization
        vlad = self.bn(vlad)
        vlad = vlad.reshape(x.size(0), x.size(1), x.size(2), -1).contiguous()  # flatten
        vlad = F.normalize(vlad, p=2, dim=3)  # L2 normalize

        return vlad

class GAFN_V2(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False, use_faiss=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(GAFN_V2, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 1.0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv1d(dim, num_clusters, kernel_size=1, bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.use_faiss = use_faiss
        self.bn = nn.BatchNorm2d(8)
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x, cluster):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=2)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x.permute(0,2,1).contiguous())
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        row_total = cluster.size(-2)
        row_sequence = np.arange(row_total)
        np.random.shuffle(row_sequence)
        centroids = nn.Parameter(cluster[:,row_sequence[0:self.num_clusters],:])
        #centroids = cluster[:,row_sequence[0:self.num_clusters],:]
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            centroids.expand(x_flatten.size(-2), -1, -1, -1).permute(1, 2, 0, 3)
        residual *= soft_assign.unsqueeze(3)
        vlad = residual.permute(0, 2, 1, 3).contiguous()#.sum(dim=-2)

        vlad = F.normalize(vlad, p=2, dim=3)  # intra-normalization
        # vlad = self.bn(vlad)
        vlad = vlad.reshape(x.size(0), x.size(1), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=2)  # L2 normalize

        return vlad

class GAFN_Four_V2(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False, use_faiss=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(GAFN_Four_V2, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 1.0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.use_faiss = use_faiss
        self.bn = nn.BatchNorm3d(8)
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x, cluster):
        N, C, D = x.shape[:3]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=3)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x.permute(0,3,1,2).contiguous())
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, D, -1)
        row_total = cluster.size(-2)
        row_sequence = np.arange(row_total)
        np.random.shuffle(row_sequence)
        #centroids = cluster[:,row_sequence[0:self.num_clusters],:]    
        centroids = nn.Parameter(cluster[:,row_sequence[0:self.num_clusters],:]) 
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1, -1).permute(1, 0, 2, 3, 4) - \
            centroids.expand(x_flatten.size(-3), -1, -1, -1).expand(x_flatten.size(-2), -1, -1, -1, -1).permute( 2, 3, 1, 0, 4)
        residual *= soft_assign.unsqueeze(4)
        vlad = residual.permute(0, 2, 3, 1, 4)#.sum(dim=-2)

        vlad = F.normalize(vlad, p=2, dim=4)  # intra-normalization
        # vlad = self.bn(vlad)
        vlad = vlad.reshape(x.size(0), x.size(1), x.size(2), -1) # flatten
        vlad = F.normalize(vlad, p=2, dim=3)  # L2 normalize

        return vlad


class GAFN_V3(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, module_dim=512, attention=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(GAFN_V3, self).__init__()
        self.dim = module_dim
        self.q_proj1 = nn.Linear(module_dim, module_dim)
        self.v_proj1 = nn.Linear(module_dim, module_dim)
        self.q_proj2 = nn.Linear(module_dim, module_dim)
        self.v_proj2 = nn.Linear(module_dim, module_dim)

        self.cat1 = nn.Linear(2 * module_dim, module_dim)
        self.cat2 = nn.Linear(2 * module_dim, module_dim)
        self.attn1 = nn.Linear(module_dim, 1)
        self.attn2 = nn.Linear(module_dim, 1)
        self.attention = attention
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.proj = nn.Sequential(  
                                    nn.Linear(2*module_dim, module_dim),
                                    nn.GELU(),
                                    nn.Dropout(p=0.1),                                          
                                )

    def forward(self, x, cluster):
        N, C, H= x.shape[:]
        row_total = cluster.size(-2)
        row_sequence = np.arange(row_total)
        np.random.shuffle(row_sequence)
        centroids =  nn.Parameter(cluster[:,row_sequence[0:C],:])
        #centroids =  cluster[:,row_sequence[0:C],:]
        if self.attention:
            x_temp = self.dropout1(x)
            q_proj = self.q_proj1(x_temp)
            v_proj = self.v_proj1(centroids)
            v_q_cat = torch.cat((v_proj, q_proj * v_proj), dim=-1)
            v_q_cat = self.cat1(v_q_cat)
            v_q_cat = self.activation1(v_q_cat)
            attn = self.attn1(v_q_cat)  # (bz, k, 1)
            attn = F.softmax(attn, dim=1)  # (bz, k, 1)
            combined_concat = (attn * centroids)

            # centroids = self.dropout2(centroids)
            # q_proj = self.q_proj2(centroids)
            # v_proj = self.v_proj2(x)
            # v_q_cat = torch.cat((v_proj, q_proj * v_proj), dim=-1)
            # v_q_cat = self.cat2(v_q_cat)
            # v_q_cat = self.activation2(v_q_cat)
            # attn = self.attn2(v_q_cat)  # (bz, k, 1)
            # attn = F.softmax(attn, dim=1)  # (bz, k, 1)
            # combined_concat2 = (attn * x)

            # combined_concat=self.proj(torch.cat((combined_concat1,combined_concat2),2))
        else:
            combined_concat = self.proj(torch.cat((x,centroids),2))
        return combined_concat

class GAFN_V3_self(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, module_dim=512, attention=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(GAFN_V3_self, self).__init__()
        self.dim = module_dim
        self.q_proj1 = nn.Linear(module_dim, module_dim)
        self.v_proj1 = nn.Linear(module_dim, module_dim)
        self.q_proj2 = nn.Linear(module_dim, module_dim)
        self.v_proj2 = nn.Linear(module_dim, module_dim)

        self.cat1 = nn.Linear(2 * module_dim, module_dim)
        self.cat2 = nn.Linear(2 * module_dim, module_dim)
        self.attn1 = nn.Linear(module_dim, 1)
        self.attn2 = nn.Linear(module_dim, 1)
        self.attention = attention
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.proj = nn.Sequential(  
                                    nn.Linear(2*module_dim, module_dim),
                                    nn.GELU(),
                                    nn.Dropout(p=0.1),                                          
                                )

    def forward(self, x, cluster):
        N, C, H= x.shape[:]
        centroids =  cluster
        #centroids =  cluster[:,row_sequence[0:C],:]
        if self.attention:
            x_temp = self.dropout1(x)
            q_proj = self.q_proj1(x_temp)
            v_proj = self.v_proj1(centroids)
            v_q_cat = torch.cat((v_proj, q_proj * v_proj), dim=-1)
            v_q_cat = self.cat1(v_q_cat)
            v_q_cat = self.activation1(v_q_cat)
            attn = self.attn1(v_q_cat)  # (bz, k, 1)
            attn = F.softmax(attn, dim=1)  # (bz, k, 1)
            combined_concat = (attn * centroids)

            # centroids = self.dropout2(centroids)
            # q_proj = self.q_proj2(centroids)
            # v_proj = self.v_proj2(x)
            # v_q_cat = torch.cat((v_proj, q_proj * v_proj), dim=-1)
            # v_q_cat = self.cat2(v_q_cat)
            # v_q_cat = self.activation2(v_q_cat)
            # attn = self.attn2(v_q_cat)  # (bz, k, 1)
            # attn = F.softmax(attn, dim=1)  # (bz, k, 1)
            # combined_concat2 = (attn * x)

            # combined_concat=self.proj(torch.cat((combined_concat1,combined_concat2),2))
        else:
            combined_concat = self.proj(torch.cat((x,centroids),2))
        return combined_concat

class GAFN_V3_cluster(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, module_dim=512, attention=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(GAFN_V3_cluster, self).__init__()
        self.dim = module_dim
        self.q_proj1 = nn.Linear(module_dim, module_dim)
        self.v_proj1 = nn.Linear(module_dim, module_dim)
        self.q_proj2 = nn.Linear(module_dim, module_dim)
        self.v_proj2 = nn.Linear(module_dim, module_dim)

        self.cat1 = nn.Linear(2 * module_dim, module_dim)
        self.cat2 = nn.Linear(2 * module_dim, module_dim)
        self.attn1 = nn.Linear(module_dim, 1)
        self.attn2 = nn.Linear(module_dim, 1)
        self.attention = attention
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.proj = nn.Sequential(  
                                    nn.Linear(2*module_dim, module_dim),
                                    nn.GELU(),
                                    nn.Dropout(p=0.1),                                          
                                )

    def forward(self, x, cluster):
        N, C, H= x.shape[:]
        row_total = cluster.size(-2)
        row_sequence = np.arange(row_total)
        np.random.shuffle(row_sequence)
        centroids =  nn.Parameter(cluster[:,row_sequence[0:C],:])
        #centroids =  cluster[:,row_sequence[0:C],:]
        if self.attention:
            centroids = self.dropout2(centroids)
            q_proj = self.q_proj2(centroids)
            v_proj = self.v_proj2(x)
            v_q_cat = torch.cat((v_proj, q_proj * v_proj), dim=-1)
            v_q_cat = self.cat2(v_q_cat)
            v_q_cat = self.activation2(v_q_cat)
            attn = self.attn2(v_q_cat)  # (bz, k, 1)
            attn = F.softmax(attn, dim=1)  # (bz, k, 1)
            combined_concat = (attn * x)

            # combined_concat=self.proj(torch.cat((combined_concat1,combined_concat2),2))
        else:
            combined_concat = self.proj(torch.cat((x,centroids),2))
        return combined_concat

class GAFN_Four_V3(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, module_dim=512, attention=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(GAFN_Four_V3, self).__init__()
        self.dim = module_dim
        self.q_proj1 = nn.Linear(module_dim, module_dim)
        self.v_proj1 = nn.Linear(module_dim, module_dim)
        self.q_proj2 = nn.Linear(module_dim, module_dim)
        self.v_proj2 = nn.Linear(module_dim, module_dim)

        self.cat1 = nn.Linear(2 * module_dim, module_dim)
        self.cat2 = nn.Linear(2 * module_dim, module_dim)
        self.attn1 = nn.Linear(module_dim, 1)
        self.attn2 = nn.Linear(module_dim, 1)
        self.attention = attention
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.proj = nn.Sequential( 
                                    nn.Linear(2*module_dim, module_dim),
                                    nn.GELU(),
                                    nn.Dropout(p=0.1),                                          
                                )

    def forward(self, x, cluster):
        N, C, H, W = x.shape[:]
        row_total = cluster.size(-2)
        row_sequence = np.arange(row_total)
        np.random.shuffle(row_sequence)
        centroids =  nn.Parameter(cluster[:,row_sequence[0:C*H],:])
        #centroids =  cluster[:,row_sequence[0:C*H],:]
        if self.attention:
            x_temp = self.dropout1(x.reshape(N,C*H,W))
            #x = x.reshape(N,C*H,W)
            q_proj = self.q_proj1(x_temp)
            v_proj = self.v_proj1(centroids)
            v_q_cat = torch.cat((v_proj, q_proj* v_proj), dim=-1)
            v_q_cat = self.cat1(v_q_cat)
            v_q_cat = self.activation1(v_q_cat)
            attn = self.attn1(v_q_cat)  # (bz, k, 1)
            attn = F.softmax(attn, dim=1)  # (bz, k, 1)
            combined_concat = (attn * centroids).reshape(N,C,H,W)

            # centroids = self.dropout2(centroids)
            # #x = x.reshape(N,C*H,W)
            # q_proj = self.q_proj2(centroids)
            # v_proj = self.v_proj2(x.reshape(N,C*H,W))
            # v_q_cat = torch.cat((v_proj, q_proj* v_proj), dim=-1)
            # v_q_cat = self.cat2(v_q_cat)
            # v_q_cat = self.activation2(v_q_cat)
            # attn = self.attn2(v_q_cat)  # (bz, k, 1)
            # attn = F.softmax(attn, dim=1)  # (bz, k, 1)
            # combined_concat2 = (attn * x.reshape(N,C*H,W)).reshape(N,C,H,W)
            # combined_concat=self.proj(torch.cat((combined_concat1,combined_concat2),3))
        else:
            combined_concat = self.proj(torch.cat((x,centroids.reshape(N,C,H,W)),3))
        return combined_concat

class GAFN_Four_V3_self(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, module_dim=512, attention=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(GAFN_Four_V3_self, self).__init__()
        self.dim = module_dim
        self.q_proj1 = nn.Linear(module_dim, module_dim)
        self.v_proj1 = nn.Linear(module_dim, module_dim)
        self.q_proj2 = nn.Linear(module_dim, module_dim)
        self.v_proj2 = nn.Linear(module_dim, module_dim)

        self.cat1 = nn.Linear(2 * module_dim, module_dim)
        self.cat2 = nn.Linear(2 * module_dim, module_dim)
        self.attn1 = nn.Linear(module_dim, 1)
        self.attn2 = nn.Linear(module_dim, 1)
        self.attention = attention
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.proj = nn.Sequential( 
                                    nn.Linear(2*module_dim, module_dim),
                                    nn.GELU(),
                                    nn.Dropout(p=0.1),                                          
                                )

    def forward(self, x, cluster):
        N, C, H, W = x.shape[:]
        centroids =  cluster.reshape(N,C*H,W)
        #centroids =  cluster[:,row_sequence[0:C*H],:]
        if self.attention:
            x_temp = self.dropout1(x.reshape(N,C*H,W))
            #x = x.reshape(N,C*H,W)
            q_proj = self.q_proj1(x_temp)
            v_proj = self.v_proj1(centroids)
            v_q_cat = torch.cat((v_proj, q_proj* v_proj), dim=-1)
            v_q_cat = self.cat1(v_q_cat)
            v_q_cat = self.activation1(v_q_cat)
            attn = self.attn1(v_q_cat)  # (bz, k, 1)
            attn = F.softmax(attn, dim=1)  # (bz, k, 1)
            combined_concat = (attn * centroids).reshape(N,C,H,W)

            # centroids = self.dropout2(centroids)
            # #x = x.reshape(N,C*H,W)
            # q_proj = self.q_proj2(centroids)
            # v_proj = self.v_proj2(x.reshape(N,C*H,W))
            # v_q_cat = torch.cat((v_proj, q_proj* v_proj), dim=-1)
            # v_q_cat = self.cat2(v_q_cat)
            # v_q_cat = self.activation2(v_q_cat)
            # attn = self.attn2(v_q_cat)  # (bz, k, 1)
            # attn = F.softmax(attn, dim=1)  # (bz, k, 1)
            # combined_concat2 = (attn * x.reshape(N,C*H,W)).reshape(N,C,H,W)
            # combined_concat=self.proj(torch.cat((combined_concat1,combined_concat2),3))
        else:
            combined_concat = self.proj(torch.cat((x,centroids.reshape(N,C,H,W)),3))
        return combined_concat

class GAFN_Four_V3_cluster(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, module_dim=512, attention=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(GAFN_Four_V3_cluster, self).__init__()
        self.dim = module_dim
        self.q_proj1 = nn.Linear(module_dim, module_dim)
        self.v_proj1 = nn.Linear(module_dim, module_dim)
        self.q_proj2 = nn.Linear(module_dim, module_dim)
        self.v_proj2 = nn.Linear(module_dim, module_dim)

        self.cat1 = nn.Linear(2 * module_dim, module_dim)
        self.cat2 = nn.Linear(2 * module_dim, module_dim)
        self.attn1 = nn.Linear(module_dim, 1)
        self.attn2 = nn.Linear(module_dim, 1)
        self.attention = attention
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.proj = nn.Sequential( 
                                    nn.Linear(2*module_dim, module_dim),
                                    nn.GELU(),
                                    nn.Dropout(p=0.1),                                          
                                )

    def forward(self, x, cluster):
        N, C, H, W = x.shape[:]
        row_total = cluster.size(-2)
        row_sequence = np.arange(row_total)
        np.random.shuffle(row_sequence)
        centroids =  nn.Parameter(cluster[:,row_sequence[0:C*H],:])
        #centroids =  cluster[:,row_sequence[0:C*H],:]
        if self.attention:
            centroids = self.dropout2(centroids)
            #x = x.reshape(N,C*H,W)
            q_proj = self.q_proj2(centroids)
            v_proj = self.v_proj2(x.reshape(N,C*H,W))
            v_q_cat = torch.cat((v_proj, q_proj* v_proj), dim=-1)
            v_q_cat = self.cat2(v_q_cat)
            v_q_cat = self.activation2(v_q_cat)
            attn = self.attn2(v_q_cat)  # (bz, k, 1)
            attn = F.softmax(attn, dim=1)  # (bz, k, 1)
            combined_concat = (attn * x.reshape(N,C*H,W)).reshape(N,C,H,W)
        else:
            combined_concat = self.proj(torch.cat((x,centroids.reshape(N,C,H,W)),3))
        return combined_concat