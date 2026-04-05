import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils import SimpleAttention, MatchingAttention, MultiHeadAttention, PositionalEncoding, SelfAttention

class DialogueRNNCell(nn.Module):

    def __init__(self, D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_att=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m_T = D_m_T
        self.D_m_A = D_m_A
        self.D_m_V = D_m_V
        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        self.g_cell_t = nn.GRUCell(D_m+D_p,D_g)
        self.p_cell_t = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell_t = nn.GRUCell(D_p,D_e)
        
        self.g_cell_a = nn.GRUCell(D_m+D_p,D_g)
        self.p_cell_a = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell_a = nn.GRUCell(D_p,D_e)
        
        self.g_cell_v = nn.GRUCell(D_m+D_p,D_g)
        self.p_cell_v = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell_v = nn.GRUCell(D_p,D_e)
        
        self.dense_t = nn.Linear(D_m_T,D_m)
        self.dense_a = nn.Linear(D_m_A,D_m)
        self.dense_v = nn.Linear(D_m_V,D_m)
        
        self.my_selfAtt1 = SelfAttention(D_g,att_type = 'general2')
        self.my_selfAtt2 = SelfAttention(D_g,att_type = 'general2')
        self.my_selfAtt3 = SelfAttention(D_g,att_type = 'general2')
        
        self.dense1 = nn.Linear(self.D_g*3,self.D_g,bias=True)
        self.dense2 = nn.Linear(self.D_g*3,self.D_g,bias=True)
        self.dense3 = nn.Linear(self.D_g*3,self.D_g,bias=True)
        
        self.selfAttention = nn.Linear(D_g,1,bias=True)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m+D_p,D_p)

        self.dropout = nn.Dropout(dropout)
        self.positional_embedding = PositionalEncoding(D_g)

        if context_attention=='simple':
            self.attention1 = SimpleAttention(D_g)
            self.attention2 = SimpleAttention(D_g)
            self.attention3 = SimpleAttention(D_g)
            
            self.attention4 = SimpleAttention(D_g)
            self.attention5 = SimpleAttention(D_g)
            
            self.attention6 = SimpleAttention(D_g)
            self.attention7 = SimpleAttention(D_g)
            
            self.attention8 = SimpleAttention(D_g)
            self.attention9 = SimpleAttention(D_g)
        else:
            
            self.attention1 = MatchingAttention(D_g, D_m, D_att, context_attention)
            self.attention2 = MatchingAttention(D_g, D_m, D_att, context_attention)
            self.attention3 = MatchingAttention(D_g, D_m, D_att, context_attention)
            
            self.attention4 = MatchingAttention(D_g, D_m, D_att, context_attention)
            self.attention5 = MatchingAttention(D_g, D_m, D_att, context_attention)
            
            self.attention6 = MatchingAttention(D_g, D_m, D_att, context_attention)
            self.attention7 = MatchingAttention(D_g, D_m, D_att, context_attention)
            
            self.attention8 = MatchingAttention(D_g, D_m, D_att, context_attention)
            self.attention9 = MatchingAttention(D_g, D_m, D_att, context_attention)

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
            # idx:z
            # j:[num_party,D_p] 2*D_p
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, Ut, Uv, Ua, qmask, g_hist_t, g_hist_v, g_hist_a, q0_t, q0_v, q0_a, e0_t, e0_v, e0_a,k=1):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e-
        """
        Ut = self.dense_t(Ut)
        Ua = self.dense_a(Ua)
        Uv = self.dense_v(Uv)

        qm_idx = torch.argmax(qmask, 1) 
        q0_sel_t = self._select_parties(q0_t, qm_idx)
        q0_sel_a = self._select_parties(q0_a, qm_idx)
        q0_sel_v = self._select_parties(q0_v, qm_idx)

        g_t = self.g_cell_t(torch.cat([Ut,q0_sel_t], dim=1),
                torch.zeros(Ut.size()[0],self.D_g).type(Ut.type()) if g_hist_t.size()[0]==0 else
                g_hist_t[-1])
        g_t = self.dropout(g_t)
        
        g_v = self.g_cell_v(torch.cat([Uv,q0_sel_v], dim=1),
                torch.zeros(Uv.size()[0],self.D_g).type(Uv.type()) if g_hist_v.size()[0]==0 else
                g_hist_v[-1])
        g_v = self.dropout(g_v)

        g_a = self.g_cell_a(torch.cat([Ua,q0_sel_a], dim=1),
                torch.zeros(Ua.size()[0],self.D_g).type(Ua.type()) if g_hist_a.size()[0]==0 else
                g_hist_a[-1])
        g_a = self.dropout(g_a)
        
        if g_hist_t.size()[0]==0:
            c_t = torch.zeros(Ut.size()[0],self.D_g).type(Ut.type())
            alpha = None
        if g_hist_a.size()[0]==0:
            c_a = torch.zeros(Ua.size()[0],self.D_g).type(Ua.type())
            alpha = None
        if g_hist_v.size()[0]==0:
            c_v = torch.zeros(Uv.size()[0],self.D_g).type(Uv.type())
            alpha = None
        else:
            g_hist_a = self.positional_embedding(g_hist_a)
            g_hist_v = self.positional_embedding(g_hist_v)
            g_hist_t = self.positional_embedding(g_hist_t)

            # Att(uT,GA) -> u_tt ; Att(uA,GA) -> u_aa ; Att(uV,GV) -> u_vv
            c_tt, alpha_tt = self.attention1(g_hist_t,Ut)
            c_vv, alpha_vv = self.attention2(g_hist_v,Uv)
            c_aa, alpha_aa = self.attention3(g_hist_a,Ua)
            
            #T & A Att(uA,GT) -> u_at ; Att(uT,GA) -> u_ta
            c_at, alpha_at = self.attention4(g_hist_t,Ua)
            c_ta, alpha_ta = self.attention5(g_hist_a,Ut)
            
            #T & V Att(uV,GT) -> u_vt ; Att(uT,GV) -> u_tv
            c_vt, alpha_vt = self.attention6(g_hist_t,Uv)
            c_tv, alpha_tv = self.attention7(g_hist_v,Ut)
            
            #A & V Att(uV,GA) -> u_va ; Att(uA,GV) -> u_av
            c_va, alpha_va = self.attention8(g_hist_a,Uv)
            c_av, alpha_av = self.attention9(g_hist_v,Ua)
            
            alpha = alpha_tt + alpha_vv + alpha_aa + alpha_ta + alpha_at + alpha_tv + alpha_vt + alpha_va + alpha_av 

            c_ttav = torch.cat([c_tt.unsqueeze(1),c_ta.unsqueeze(1),c_tv.unsqueeze(1)],1) # batch, 3, D_g
            c_aatv = torch.cat([c_aa.unsqueeze(1),c_at.unsqueeze(1),c_av.unsqueeze(1)],1)
            c_vvta = torch.cat([c_vv.unsqueeze(1),c_vt.unsqueeze(1),c_va.unsqueeze(1)],1)
            
            c_t, _ = self.my_selfAtt1(c_ttav) # batch, D_g * 3
            c_t = self.dense1(c_t)
            
            c_a, _ = self.my_selfAtt2(c_aatv)
            c_a = self.dense2(c_a)
            
            c_v, _ = self.my_selfAtt3(c_vvta)
            c_v = self.dense3(c_v)

        q_t, e_t = self.rnn_cell(Ut,c_t,qmask,qm_idx,q0_t,e0_t,self.p_cell_t,self.e_cell_t)
        q_a, e_a = self.rnn_cell(Ua,c_a,qmask,qm_idx,q0_a,e0_a,self.p_cell_a,self.e_cell_a)
        q_v, e_v = self.rnn_cell(Uv,c_v,qmask,qm_idx,q0_v,e0_v,self.p_cell_v,self.e_cell_v)
        
        return g_t,q_t,e_t,g_v,q_v,e_v,g_a,q_a,e_a,alpha
        
class DialogueRNN(nn.Module):

    def __init__(self, D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_att=100, dropout=0.5):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e,
                            listener_state, context_attention, D_att, dropout)
        self.selfAttention = nn.Linear(D_e,1,bias=True)

    def forward(self, Ut, Uv, Ua, qmask, train=False):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist_t = torch.zeros(0).type(Ut.type()) # 0-dimensional tensor
        g_hist_v = torch.zeros(0).type(Uv.type()) # 0-dimensional tensor
        g_hist_a = torch.zeros(0).type(Ua.type()) # 0-dimensional tensor
        q_t = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(Ut.type()) # batch, party, D_p
        e_t = torch.zeros(0).type(Ut.type()) # batch, D_e
        et = e_t
        
        q_v = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(Uv.type()) # batch, party, D_p
        e_v = torch.zeros(0).type(Uv.type()) # batch, D_e
        ev = e_v

        q_a = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(Ua.type()) # batch, party, D_p
        e_a = torch.zeros(0).type(Ua.type()) # batch, D_e
        ea = e_a

        alpha = []
        for u_t, u_v, u_a, qmask_ in zip(Ut, Uv, Ua, qmask):

            g_t,q_t,e_t,g_v,q_v,e_v,g_a,q_a,e_a,alpha_ = self.dialogue_cell(u_t, u_v, u_a, qmask_, g_hist_t, g_hist_v, g_hist_a, q_t,q_v,q_a, e_t,e_v,e_a,k=5)
            
            g_hist_t = torch.cat([g_hist_t, g_t.unsqueeze(0)],0)
            g_hist_v = torch.cat([g_hist_v, g_v.unsqueeze(0)],0)
            g_hist_a = torch.cat([g_hist_a, g_a.unsqueeze(0)],0)

            et = torch.cat([et, e_t.unsqueeze(0)],0)
            ev = torch.cat([ev, e_v.unsqueeze(0)],0)
            ea = torch.cat([ea, e_a.unsqueeze(0)],0)

            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        e = torch.cat([et, ev, ea],dim = -1) # seq_len, batch, De*3
        score_t = torch.cat([et.unsqueeze(2), ev.unsqueeze(2), ea.unsqueeze(2)],dim = 2) # seq_len, batch, 3, De
        score_t = self.selfAttention(score_t) # seq_len, batch, 3, 1
        score_t = F.softmax(score_t,dim=-1)
        score_t = score_t.squeeze(-1)     # seq_len, batch, 3
        score_t = score_t.repeat(1,1,1,self.D_e).view(len(e),len(e[0]),-1)  # seq_len, batch_size, De*3

        e = torch.mul(e,score_t)  # seq_len, batch_size, De*3

        return e,alpha # seq_len, batch, D_e*3


class BiModel_triple(nn.Module): 

    def __init__(self, D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, D_h, 
            n_classes=7, listener_state=False, context_attention='simple', D_att=100, dropout=0.5):
        super(BiModel_triple, self).__init__()

        self.D_m         = D_m
        self.D_g         = D_g
        self.D_p         = D_p
        self.D_e         = D_e
        self.D_h         = D_h
        self.n_classes = n_classes
        self.dropout     = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout)

        self.dialog_rnn_f = DialogueRNN(D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_att, dropout)
        
        self.dialog_rnn_r = DialogueRNN(D_m_T, D_m_V, D_m_A,D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_att, dropout)
        self.positional_embedding = PositionalEncoding(D_e*6)

        self.linear      = nn.Linear(6*D_e, 3*D_h)
        self.smax_fc     = nn.Linear(3*D_h, n_classes)
        self.matchatt = MatchingAttention(6*D_e,6*D_e,att_type='general')
        self.multiheadAttn = MultiHeadAttention(6*D_e, 6*D_e, 4)

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

    def forward(self, Ut, Uv, Ua, qmask, umask, train=False, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions_f, alpha_f = self.dialog_rnn_f(Ut, Uv, Ua, qmask, train) # seq_len, batch, D_e

        emotions_f = self.dropout_rec(emotions_f)  # seq_len, batch_size, De*3
        rev_Ut = self._reverse_seq(Ut, umask)
        rev_Uv = self._reverse_seq(Uv, umask)
        rev_Ua = self._reverse_seq(Ua, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_r, alpha_r  = self.dialog_rnn_r(rev_Ut, rev_Uv, rev_Ua, rev_qmask)
        emotions_r = self._reverse_seq(emotions_r, umask)
        emotions_r = self.dropout_rec(emotions_r)  # seq_len, batch_size, De*3
        emotions = torch.cat([emotions_f,emotions_r],dim=-1) # seq_len, batch_size, De*6
        emotions = self.positional_embedding(emotions) # seq_len, batch_size, De*6


        if att2:
            # MultiHeadAttention
            att_emotions, alpha = self.multiheadAttn(emotions, emotions, emotions)
            alpha = list(alpha)
            # att_emotions : e'=[e1',e2',...,en'] 
            hidden = F.relu(self.linear(att_emotions))  # seq_len, batch_size, Dh*3
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2) # seq_len, batch, n_classes
        log_prob = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])

        return log_prob