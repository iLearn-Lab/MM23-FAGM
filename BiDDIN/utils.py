import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M) # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector

        return attn_pool, alpha

class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim

        return attn_pool, alpha

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, attn_mask=None):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask:
            scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, cand_dim, embed_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.transform = nn.Linear(cand_dim, embed_dim, bias=False)
        assert embed_dim % n_heads == 0
        self.cand_dim = cand_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dim_per_head = self.embed_dim // self.n_heads
        self.W_Q = nn.Linear(self.embed_dim, self.dim_per_head * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.embed_dim, self.dim_per_head * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.embed_dim, self.dim_per_head * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.dim_per_head, self.embed_dim, bias=False)

    def forward(self, input_Q, input_K, input_V):
        # input_Q: [seq_len, batch_size, cand_dim]  or [batch_size, cand_dim]
        # input_K: [seq_len, batch_size, embed_dim]
        # input_V: [seq_len, batch_size, embed_dim]
        if len(input_Q.shape) == 2:
            input_Q = input_Q.unsqueeze(0)
        if input_Q.size(-1) != input_K.size(-1):
            input_Q = self.transform(input_Q)   # [1/seq_len, batch_size, embed_dim]
        residual = input_Q  # [1/seq_len, batch_size, embed_dim]
        batch_size = input_Q.size(1)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1,2) # [batch_size, n_heads, 1/seq_len, dim_per_head]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1,2) # [batch_size, n_heads, seq_len, dim_per_head]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1,2) # [batch_size, n_heads, seq_len, dim_per_head]

        context, attn = ScaledDotProductAttention(self.dim_per_head)(Q, K, V)
        # context: [batch_size, n_heads, 1/seq_len, dim_per_head]
        # attn: [batch_size, n_heads, 1/seq_len, seq_len]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.dim_per_head).transpose(0, 1) # [1/seq_len, batch_size, embed_size]
        output = self.fc(context) # [1/seq_len, batch_size, embed_size]
        attn = attn.reshape(attn.size(2), -1, attn.size(3))  # [1/seq_len, batch_size, seq_len]
        return nn.LayerNorm(self.embed_dim).cuda()(output + residual), attn

class SelfAttention(nn.Module):

    def __init__(self, input_dim, att_type='general'):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.att_type = att_type
        self.scalar = nn.Linear(self.input_dim,1,bias=True)

    def forward(self, M, x=None):
        """
        now M -> (batch, seq_len, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        if self.att_type == 'general':
            scale = self.scalar(M) # seq_len, batch, 1
            alpha = F.softmax(scale, dim=0).permute(0,2,1) # batch, 1, seq_len
            attn_pool = torch.bmm(alpha, M)[:,0,:] # batch, vector/input_dim
        if self.att_type == 'general2':
            scale = self.scalar(M) # seq_len, batch, 1
            alpha = F.softmax(scale, dim=0).permute(0,2,1) # batch, 1, seq_len
            att_vec_bag = []
            for i in range(M.size()[1]):
                alp = alpha[:,:,i]
                vec = M[:, i, :]
                alp = alp.repeat(1,self.input_dim)
                att_vec = torch.mul(alp, vec) # batch, vector/input_dim
                att_vec = att_vec + vec
                att_vec_bag.append(att_vec)
            attn_pool = torch.cat(att_vec_bag, -1)

        return attn_pool, alpha

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        self.dropout = nn.Dropout(dropout)
        X = torch.arange(max_len, dtype=torch.float32).reshape(\
            -1, 1) / torch.pow(10000, torch.arange(\
                0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    
    def forward(self, X):
        
        # X: seq_len, batch_size, num_hiddens
        X = X.transpose(0, 1)
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X.transpose(0, 1))

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss
