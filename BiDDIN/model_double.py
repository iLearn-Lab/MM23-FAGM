import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils import SimpleAttention, MatchingAttention, MultiHeadAttention, PositionalEncoding, SelfAttention

class DialogueRNNCell(nn.Module):

    def __init__(self, D_m_A, D_m_B, D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_att=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m_A = D_m_A
        self.D_m_B = D_m_B
        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state

        self.g_cell_a = nn.GRUCell(D_m+D_p,D_g)
        self.p_cell_a = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell_a = nn.GRUCell(D_p,D_e)

        self.g_cell_b = nn.GRUCell(D_m+D_p,D_g)
        self.p_cell_b = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell_b = nn.GRUCell(D_p,D_e)

        self.dense_a = nn.Linear(D_m_A,D_m)
        self.dense_b = nn.Linear(D_m_B,D_m)

        self.my_self_Att1 = SelfAttention(D_g,att_type = 'general2')
        self.my_self_Att2 = SelfAttention(D_g,att_type = 'general2')
        self.dense1 = nn.Linear(self.D_g*2,self.D_g,bias=True)
        self.dense2 = nn.Linear(self.D_g*2,self.D_g,bias=True)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m+D_p,D_p)

        self.dropout = nn.Dropout(dropout)
        self.positional_embedding = PositionalEncoding(D_g)
        
        if context_attention=='simple':
            
            self.attention = SimpleAttention(D_g)
        else:
            
            self.attention = MatchingAttention(D_g, D_m, D_att, context_attention)
        
        if context_attention=='simple':
            self.attention1 = SimpleAttention(D_g)
            self.attention2 = SimpleAttention(D_g)
            self.attention3 = SimpleAttention(D_g)
            self.attention4 = SimpleAttention(D_g)
        else:
            self.attention1 = MatchingAttention(D_g, D_m, D_att, context_attention)
            self.attention2 = MatchingAttention(D_g, D_m, D_att, context_attention)
            self.attention3 = MatchingAttention(D_g, D_m, D_att, context_attention)
            self.attention4 = MatchingAttention(D_g, D_m, D_att, context_attention)

    def rnn_cell(self,U,c_,qmask,qm_idx,q0,e0,p_cell,e_cell):
        U_c_ = torch.cat([U,c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_ = p_cell(U_c_.contiguous().view(-1,self.D_m+self.D_g),
                q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
        qs_ = self.dropout(qs_)

        if self.listener_state:
            U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                    expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_p)
            U_ss_ = torch.cat([U_,ss_],1)
            ql_ = self.l_cell(U_ss_,q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
            ql_ = self.dropout(ql_)
        else:
            ql_ = q0
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_*(1-qmask_) + qs_*qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0\
                else e0
        e_ = e_cell(self._select_parties(q_,qm_idx), e0)
        e_ = self.dropout(e_)
        return q_,e_

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, Ua, Ub, qmask, g_hist_a, g_hist_b, q0_a, q0_b, e0_a, e0_b,k=1):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """

        Ua = self.dense_a(Ua)
        Ub = self.dense_b(Ub)

        qm_idx = torch.argmax(qmask, 1)
        q0_sel_a = self._select_parties(q0_a, qm_idx)
        q0_sel_b = self._select_parties(q0_b, qm_idx)

        g_a = self.g_cell_a(torch.cat([Ua,q0_sel_a], dim=1),
                torch.zeros(Ua.size()[0],self.D_g).type(Ua.type()) if g_hist_a.size()[0]==0 else
                g_hist_a[-1])
        g_a = self.dropout(g_a)

        g_b = self.g_cell_b(torch.cat([Ub,q0_sel_b], dim=1),
                torch.zeros(Ub.size()[0],self.D_g).type(Ub.type()) if g_hist_b.size()[0]==0 else
                g_hist_b[-1])
        g_b = self.dropout(g_b)

        if g_hist_a.size()[0]==0:
            c_a = torch.zeros(Ua.size()[0],self.D_g).type(Ua.type())
            alpha = None
        if g_hist_b.size()[0]==0:
            c_b = torch.zeros(Ub.size()[0],self.D_g).type(Ub.type())
            alpha = None
        else:
            g_hist_a = self.positional_embedding(g_hist_a)
            g_hist_b = self.positional_embedding(g_hist_b)

            c_aa, alpha_aa = self.attention1(g_hist_a,Ua)
            c_ba, alpha_ba = self.attention2(g_hist_a,Ub)
            c_ab, alpha_ab = self.attention3(g_hist_b,Ua)
            c_bb, alpha_bb = self.attention4(g_hist_b,Ub)
            
            alpha = alpha_aa + alpha_ab + alpha_ba + alpha_bb

            c_aab = torch.cat([c_aa.unsqueeze(1),c_ab.unsqueeze(1)],1)
            c_bba = torch.cat([c_bb.unsqueeze(1),c_ba.unsqueeze(1)],1)
            c_a, _ = self.my_self_Att1(c_aab)
            c_a = self.dense1(c_a)
            c_b, _ = self.my_self_Att2(c_bba)
            c_b = self.dense2(c_b)
        
        q_a, e_a = self.rnn_cell(Ua,c_a,qmask,qm_idx,q0_a,e0_a,self.p_cell_a,self.e_cell_a)
        q_b, e_b = self.rnn_cell(Ub,c_b,qmask,qm_idx,q0_b,e0_b,self.p_cell_b,self.e_cell_b)
        
        return g_a,q_a,e_a,g_b,q_b,e_b,alpha

class DialogueRNN(nn.Module):

    def __init__(self, D_m_A, D_m_B,D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_att=100, dropout=0.5):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(D_m_A, D_m_B, D_m, D_g, D_p, D_e,
                            listener_state, context_attention, D_att, dropout)
        self.self_Attention = nn.Linear(D_e,1,bias=True)

    def forward(self, Ua, Ub, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist_a = torch.zeros(0).type(Ua.type()) # 0-dimensional tensor
        g_hist_b = torch.zeros(0).type(Ub.type()) # 0-dimensional tensor
        q_a = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(Ua.type()) # batch, party, D_p
        e_a = torch.zeros(0).type(Ua.type()) # batch, D_e
        ea = e_a

        q_b = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(Ub.type()) # batch, party, D_p
        e_b = torch.zeros(0).type(Ub.type()) # batch, D_e
        eb = e_b

        alpha = []
        for u_a,u_b,qmask_ in zip(Ua,Ub, qmask):
            g_a,q_a,e_a,g_b,q_b,e_b,alpha_ = self.dialogue_cell(u_a, u_b, qmask_, g_hist_a, g_hist_b, q_a, q_b, e_a, e_b, k=5)
            g_hist_a = torch.cat([g_hist_a, g_a.unsqueeze(0)],0)
            g_hist_b = torch.cat([g_hist_b, g_b.unsqueeze(0)],0)
            ea = torch.cat([ea, e_a.unsqueeze(0)],0)
            eb = torch.cat([eb, e_b.unsqueeze(0)],0)
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        e = torch.cat([ea, eb],dim = -1)
        score_a = torch.cat([ea.unsqueeze(2), eb.unsqueeze(2)],dim = 2)
        score_a = self.self_Attention(score_a)
        score_a = F.softmax(score_a,dim=-1)
        score_a = score_a.repeat(1,1,1,self.D_e).view(len(e),len(e[0]),-1)

        e = torch.mul(e,score_a)

        return e,alpha # [seq_len, batch, D_e]

class BiModel_double(nn.Module):

    def __init__(self, D_m_A,D_m_B,D_m, D_g, D_p, D_e, D_h,
                 n_classes=7, listener_state=False, context_attention='simple', D_att=100, dropout=0.5):
        super(BiModel_double, self).__init__()

        self.D_m_A     = D_m_A
        self.D_m_B     = D_m_B
        self.D_m       = D_m
        self.D_g       = D_g
        self.D_p       = D_p
        self.D_e       = D_e
        self.D_h       = D_h
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout)
        self.dialog_rnn_f = DialogueRNN(D_m_A,D_m_B,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_att, dropout)
        self.dialog_rnn_r = DialogueRNN(D_m_A,D_m_B,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_att, dropout)
        self.positional_embedding1 = PositionalEncoding(D_e*2)
        self.positional_embedding = PositionalEncoding(D_e*4)

        self.linear     = nn.Linear(4*D_e, 2*D_h)
        self.smax_fc    = nn.Linear(2*D_h, n_classes)
        self.multihead_Attn = MultiHeadAttention(4*D_e,4*D_e,4)
        self.matchatt = MatchingAttention(4*D_e,4*D_e,att_type='general2')

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, Ua, Ub, qmask, umask,att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions_f, alpha_f = self.dialog_rnn_f(Ua,Ub, qmask) # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)
        rev_Ua = self._reverse_seq(Ua, umask)
        rev_Ub = self._reverse_seq(Ub, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_r, alpha_r = self.dialog_rnn_r(rev_Ua,rev_Ub, rev_qmask)
        emotions_r = self._reverse_seq(emotions_r, umask)
        emotions_r = self.dropout_rec(emotions_r)
        emotions = torch.cat([emotions_f,emotions_r],dim=-1)
        emotions = self.positional_embedding(emotions) # seq_len, batch_size, De*4
        if att2:
            # MultiHeadAttention
            att_emotions, alpha = self.multihead_Attn(emotions, emotions, emotions)
            # att_emotions : e'=[e1',e2',...,en'] 
            hidden = F.relu(self.linear(att_emotions))  # seq_len, batch_size, Dh*3
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2) # seq_len, batch, n_classes
        log_prob = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])


        return log_prob
