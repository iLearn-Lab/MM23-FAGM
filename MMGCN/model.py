import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, math
from model_GCN import TextCNN
from model_mm import MM_GCN


class FocalLoss(nn.Module):
    
    def __init__(self, gamma = 2, alpha = 0.75, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)


        log_p = F.log_softmax(logits,dim=-1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    batch_size = features.size(1)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.cuda()

    return node_features

class DialogueGCNModel(nn.Module):

    def __init__(self, D_m, D_e, graph_hidden_size, n_speakers, window_past, window_future,
                 n_classes=7, dropout=0.5, no_cuda=False, alpha=0.2,use_residue=True,
                 D_m_v=512,D_m_a=100,modals='avl',att_type='gated',av_using_lstm=False, dataset='IEMOCAP',
                 use_speaker=True, use_modal=False, multi_modal=True):
        
        super(DialogueGCNModel, self).__init__()

        self.multi_modal=multi_modal
        self.no_cuda = no_cuda
        self.alpha = alpha
        self.dropout = dropout
        self.use_residue = use_residue
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        if self.att_type == 'gated' or self.att_type == 'concat_subsequently':
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.use_bert_seq = False
        self.dataset = dataset
       
        if not self.multi_modal:
            if len(self.modals) == 3:
                hidden_ = 250
            elif ''.join(self.modals) == 'al':
                hidden_ = 150
            elif ''.join(self.modals) == 'vl':
                hidden_ = 150
            else:
                hidden_ = 100
            self.linear_ = nn.Linear(D_m, hidden_)
            self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        else:
            if 'a' in self.modals:
                hidden_a = 200
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                if self.av_using_lstm:
                    self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'v' in self.modals:
                hidden_v = 200
                self.linear_v = nn.Linear(D_m_v, hidden_v)
                if self.av_using_lstm:
                    self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'l' in self.modals:
                hidden_l = 200 
                if self.use_bert_seq:
                    self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                else:
                    self.linear_l = nn.Linear(D_m, hidden_l)
                self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)


        self.window_past = window_past
        self.window_future = window_future

        self.graph_model = MM_GCN(a_dim=2*D_e, v_dim=2*D_e, l_dim=2*D_e, n_dim=2*D_e, nlayers=4, nhidden=graph_hidden_size, nclass=n_classes, 
                                  dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True, return_feature=self.return_feature, 
                                  use_residue=self.use_residue, n_speakers=n_speakers, modals=self.modals, 
                                  use_speaker=self.use_speaker, use_modal=self.use_modal)
            
        edge_type_mapping = {} 
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping
        self.dropout_ = nn.Dropout(self.dropout)
        if self.multi_modal:    
            if self.att_type == 'concat_subsequently':
                self.smax_fc = nn.Linear(300 * len(self.modals), n_classes) if self.use_residue else nn.Linear(100 * len(self.modals), n_classes)
            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    self.smax_fc = nn.Linear(100*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear(100, n_classes)
            else:
                self.smax_fc = nn.Linear(2*D_e+graph_hidden_size*len(self.modals), n_classes)
        else:
            self.smax_fc = nn.Linear(2*D_e+graph_hidden_size*len(self.modals), n_classes)

    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, seq_lengths, U_a=None, U_v=None):
        
        if not self.multi_modal:
            U = self.linear_(U)
            emotions, hidden = self.lstm(U)
        else:
            if 'a' in self.modals:
                U_a = self.linear_a(U_a)
                if self.av_using_lstm:
                    emotions_a, hidden_a = self.lstm_a(U_a)
                else:
                    emotions_a = U_a
            if 'v' in self.modals:
                U_v = self.linear_v(U_v)
                if self.av_using_lstm:
                    emotions_v, hidden_v = self.lstm_v(U_v)
                else:
                    emotions_v = U_v
            if 'l' in self.modals:
                if self.use_bert_seq:
                    U_ = U.reshape(-1,U.shape[-2],U.shape[-1])
                    U = self.txtCNN(U_).reshape(U.shape[0],U.shape[1],-1)
                else:
                    U = self.linear_l(U)
                emotions_l, hidden_l = self.lstm_l(U)

        if not self.multi_modal:
            features = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
        else:
            if 'a' in self.modals:
                features_a = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
            else:
                features_a = []

            if 'v' in self.modals:
                features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            else:
                features_v = []

            if 'l' in self.modals:
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
            else:
                features_l = []

        # MMGCN
        if self.multi_modal:
            emotions_feat = self.graph_model(features_a, features_v, features_l, seq_lengths, qmask)
        else:
            emotions_feat = self.graph_model(features, [], [], seq_lengths, qmask)
            
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)
        log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)

        
        return log_prob