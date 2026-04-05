import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,  classification_report
from utils import MaskedNLLLoss
from dataloader import get_IEMOCAP_loaders, get_MELD_loaders
import numpy as np

from model_single import BiModel_single
from model_double import BiModel_double
from model_triple import BiModel_triple

ACTS = {}

def hook(module, args, output):
    ACTS.update({module: args[0].detach()})

def get_act():
    act = ACTS.copy()
    ACTS.clear()
    return act

MODAL_SPEC = {'text':[], 'audio':[], 'visual':[]}
MODAL_GEN = []
ISCORE = {}
PEN = {}

def modulation_init(model, dataloader, cuda_flag):
    model.eval()
    with torch.no_grad():
        handles = []
        for name, module in model.named_modules():
            if not list(module.children()) and 'smax_fc' not in name:
                if list(module.parameters()) and not isinstance(module, torch.nn.Embedding):
                    handles.append(module.register_forward_hook(hook))

        data = next(iter(dataloader))
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
    

        model(textf, visuf, acouf, qmask, umask)
        all_act = get_act()

        model(torch.zeros_like(textf), visuf, torch.zeros_like(acouf), qmask, umask)
        v_act = get_act()

        model(torch.zeros_like(textf), torch.zeros_like(visuf), acouf, qmask, umask)
        a_act = get_act()

        model(textf, torch.zeros_like(visuf), torch.zeros_like(acouf), qmask, umask)
        l_act = get_act()

        for key in all_act.keys():
            if torch.sum(all_act[key]-l_act[key]) == 0:
                MODAL_SPEC['text'].append(key)
            if torch.sum(all_act[key]-a_act[key]) == 0:
                MODAL_SPEC['audio'].append(key)
            if torch.sum(all_act[key]-v_act[key]) == 0:
                MODAL_SPEC['visual'].append(key)
        
        
        spec = MODAL_SPEC['text'] + MODAL_SPEC['audio'] + MODAL_SPEC['visual']
        for module in all_act.keys():
            if module not in spec:
                MODAL_GEN.append(module)
        
        for handle in handles:
            handle.remove()

def get_penalty(tensor, rate):
    pos_num = torch.sum((tensor>0).int()).item()
    sorted = torch.sort(tensor, descending=True)[0]
    threshold = sorted[max(int(pos_num*rate)-1, 0)]
    penalty = (tensor > threshold).float()
    return penalty

def modulation(model, all_prob, textf, qmask, umask, acouf, visuf, label, step, args):
    model.eval()
    if step % args.tau == 0:
        with torch.no_grad():
            all_act = get_act()
            loss_all = loss_function(all_prob, label, umask)
            
            el_prob = model(torch.zeros_like(textf), visuf, acouf, qmask, umask)
            loss_el = loss_function(el_prob, label, umask)
            el_act = get_act()

            ev_prob = model(textf, torch.zeros_like(visuf), acouf, qmask, umask)
            loss_ev = loss_function(ev_prob, label, umask)
            ev_act = get_act()

            ea_prob = model(textf, visuf, torch.zeros_like(acouf), qmask, umask)
            loss_ea = loss_function(ea_prob, label, umask)
            ea_act = get_act()
            
            score = torch.tensor([loss_ea-loss_all, loss_ev-loss_all, loss_el-loss_all])
            ratio = F.softmax(0.1*score, dim=0)
            r_min, _ = torch.min(ratio, dim=0)
            iscore = (ratio - r_min)**args.gamma

            ISCORE.update({"audio": iscore[0]})
            ISCORE.update({"video": iscore[1]})
            ISCORE.update({"text": iscore[2]})

    for module in MODAL_SPEC['audio']:
        for param in module.parameters():
            if param.grad is not None:
                param.grad *= (1 - ISCORE["audio"])

    for module in MODAL_SPEC['visual']:
        for param in module.parameters():
            if param.grad is not None:
                param.grad *= (1 - ISCORE["video"])
                
    for module in MODAL_SPEC['text']:
        for param in module.parameters():
            if param.grad is not None:
                param.grad *= (1 - ISCORE["text"])
        
    for module in MODAL_GEN:
        if step % args.tau == 0:
            delta_l = torch.abs(all_act[module] - el_act[module])
            delta_l = torch.mean(delta_l.reshape(-1, delta_l.size(-1)), dim=0)
            delta_a = torch.abs(all_act[module] - ea_act[module])
            delta_a = torch.mean(delta_a.reshape(-1, delta_a.size(-1)), dim=0)
            delta_v = torch.abs(all_act[module] - ev_act[module])
            delta_v = torch.mean(delta_v.reshape(-1, delta_v.size(-1)), dim=0)

            # rate = (epoch/args.epochs)**args.beta
            rate = args.beta

            delta = delta_a - torch.max(torch.stack([delta_l, delta_v], dim=0), dim=0)[0]
            pen_a = get_penalty(delta, rate)
            pen_a = pen_a * ISCORE["audio"]

            delta = delta_v - torch.max(torch.stack([delta_a, delta_l], dim=0), dim=0)[0]
            pen_v = get_penalty(delta, rate)
            pen_v = pen_v * ISCORE["video"]
        
            delta_l = delta_l - torch.max(torch.stack([delta_a, delta_v], dim=0), dim=0)[0]
            pen_l = get_penalty(delta, rate)
            pen_l = pen_l * ISCORE["text"]

            pen = 1-(pen_a + pen_v + pen_l)
            PEN.update({module: pen})

        for param in module.parameters():
            if param.grad is not None:
                if len(param.grad.size()) > 1: # weight
                    if module.__class__.__name__ == 'GRUCell' and param.grad.size()[1] != PEN[module].size():
                        continue
                    param.grad *= PEN[module].unsqueeze(0)
    model.train()


def train_or_eval_model(model, loss_function, dataloader, epoch=0, args=None, optimizer=None, train=False):
    
    losses, preds, labels, masks = [], [], [], []
    assert not train or optimizer != None

    if train:
        model.train()
    else:
        model.eval()

    step = 0
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        handles = []
        if train == True and args.modulation == True and step % args.tau == 0:
            for module in MODAL_GEN:
                handles.append(module.register_forward_hook(hook))
        
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        if args.modals == 'tva':
            log_prob = model(textf, visuf, acouf, qmask, umask)  # seq_len, batch, n_classes
        elif args.modals == 'tv':
            log_prob = model(textf, visuf, qmask, umask)  # seq_len, batch, n_classes
        elif args.modals == 'ta':
            log_prob = model(textf, acouf, qmask, umask)  # seq_len, batch, n_classes
        elif args.modals == 'va':
            log_prob = model(visuf, acouf, qmask, umask)  # seq_len, batch, n_classes
        elif args.modals == 't':
            log_prob = model(textf, qmask, umask)  # seq_len, batch, n_classes
        elif args.modals == 'v':
            log_prob = model(visuf, qmask, umask)  # seq_len, batch, n_classes
        elif args.modals == 'a':
            log_prob = model(acouf, qmask, umask)  # seq_len, batch, n_classes
        
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(log_prob, labels_, umask)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(labels_.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item() * masks[-1].sum())

        if train:
            loss.backward()
    
            if args.modulation == True:

                modulation(model, log_prob, textf, qmask, umask, acouf, visuf, labels_, step, args)
                for handle in handles:
                    handle.remove()

            optimizer.step()
            step += 1

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')
    
    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--modals', default='tva', help='modals to fusion: tva')
    parser.add_argument('--dataset', default='IEMOCAP', help='IEMOCAP/MELD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=30, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='class weight')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--output', default='./outputs', help='Model save path')
    parser.add_argument('--name', default='demo', help='Experiment name')
    parser.add_argument('--test', action='store_true', default=False, help='Test label')
    parser.add_argument('--test_modal', default='t', help='Test label')
    parser.add_argument('--load', default='', help='Load model path')
    parser.add_argument('--data_dir', type=str, default='./data/IEMOCAP_features/IEMOCAP_features_raw.pkl', help='dataset dir')
    parser.add_argument('--log_dir', type=str, default='log/', help='tensorboard save path')
    parser.add_argument('--beta', type=float, default=1, help='')
    parser.add_argument('--gamma', type=float, default=1, help='')
    parser.add_argument('--tau', type=float, default=1, help='')
    parser.add_argument('--modulation', action='store_true', default=False, help='Enables grad modulation')

    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    batch_size = args.batch_size
    cuda = args.cuda
    n_epochs = args.epochs

    if args.dataset == 'IEMOCAP':
        n_classes = 6
        D_m_T = 100
        D_m_V = 512
        D_m_A = 100
    elif args.dataset == 'MELD':
        n_classes = 7
        D_m_T = 600
        D_m_V = 342
        D_m_A = 300
    else:
        raise ValueError('There is no such dataset')

    D_m = D_m_T
    D_g = 150 # D_g: global context size vector
    D_p = 150 # D_p: party's state
    D_e = 100 # D_e: emotion's represent
    D_h = 100 # D_h: linear's emotion's represent
    D_a = 100  # concat attention  (alpha dim)

    if args.modals == 'tva':
        model = BiModel_triple(D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, D_h, 
                               n_classes=n_classes, listener_state=args.active_listener, 
                               context_attention=args.attention, dropout=args.dropout)
    elif args.modals == 'tv':
        model = BiModel_double(D_m_T, D_m_V, D_m, D_g, D_p, D_e, D_h, 
                               n_classes=n_classes, listener_state=args.active_listener, 
                               context_attention=args.attention, dropout=args.dropout)
    elif args.modals == 'ta':
        model = BiModel_double(D_m_T, D_m_A, D_m, D_g, D_p, D_e, D_h, 
                               n_classes=n_classes, listener_state=args.active_listener, 
                               context_attention=args.attention, dropout=args.dropout)
    elif args.modals == 'va':
        model = BiModel_double(D_m_V, D_m_A, D_m, D_g, D_p, D_e, D_h, 
                               n_classes=n_classes, listener_state=args.active_listener, 
                               context_attention=args.attention, dropout=args.dropout)
    elif args.modals == 't':
        model = BiModel_single(D_m_T, D_g, D_p, D_e, D_h, 
                               n_classes=n_classes, listener_state=args.active_listener, 
                               context_attention=args.attention, dropout=args.dropout)
    elif args.modals == 'v':
        model = BiModel_single(D_m_V, D_g, D_p, D_e, D_h, 
                               n_classes=n_classes, listener_state=args.active_listener, 
                               context_attention=args.attention, dropout=args.dropout)
    elif args.modals == 'a':
        model = BiModel_single(D_m_A, D_g, D_p, D_e, D_h, 
                               n_classes=n_classes, listener_state=args.active_listener, 
                               context_attention=args.attention, dropout=args.dropout)

    if cuda:
        model.cuda()

    if args.dataset == 'IEMOCAP':
        target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
        loss_weights = torch.FloatTensor([1 / 0.086747, 1 / 0.144406, 1 / 0.227883, 1 / 0.160585, 1 / 0.127711, 1 / 0.252668])
    elif args.dataset == 'MELD':
        target_names = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
        loss_weights = torch.FloatTensor([1.0 / 0.466750766, 1.0 / 0.122094071, 1.0 / 0.027752748, 1.0 / 0.071544422, 
                                        1.0 / 0.171742656, 1.0 / 0.026401153, 1.0 / 0.113714183])
    else:
        raise ValueError('There is no such dataset')
    
    if args.class_weight:    
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader =  get_IEMOCAP_loaders(args.data_dir,
                                                                    valid=0.1, batch_size=batch_size, num_workers=2)
    elif args.dataset == 'MELD':
        train_loader, valid_loader, test_loader =  get_MELD_loaders(args.data_dir,
                                                                valid=0.1, batch_size=batch_size, num_workers=2)
    else:
        raise ValueError('There is no such dataset')
        
    if args.test == False:
        best_fscore = None
        counter = 0
        if args.modulation:
            modulation_init(model, train_loader, cuda)

        for e in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, train_loader, e, args, optimizer, True)
 
            end_time = time.time()
            train_time = round(end_time-start_time, 2)

            start_time = time.time()
            with torch.no_grad():
                valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, valid_loader, e, args)
            
            end_time = time.time()
            valid_time = round(end_time-start_time, 2)

            if args.tensorboard:
                writer.add_scalar('val/accuracy', valid_acc, e)
                writer.add_scalar('val/fscore', valid_fscore, e)
                writer.add_scalar('val/loss', valid_loss, e)
                writer.add_scalar('train/accuracy', train_acc, e)
                writer.add_scalar('train/fscore', train_fscore, e)
                writer.add_scalar('train/loss', train_loss, e)

            print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, train_time: {} sec, valid_time: {} sec'. \
                format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, train_time, valid_time))

            if best_fscore == None:
                best_fscore = valid_fscore
            elif valid_fscore > best_fscore:
                best_fscore = valid_fscore
                counter = 0
                path = os.path.join(args.output, args.dataset, args.modals)
                if not os.path.isdir(path): os.makedirs(path)
                torch.save(model.state_dict(), os.path.join(path, args.name+'.pth'))
            else:
                counter += 1
                # if counter >= 10:
                #     print("Early stopping")
                #     break

        if args.tensorboard:
            writer.close()
        
    if args.test == True:
        model.load_state_dict(torch.load(args.load))
    else:
        model.load_state_dict(torch.load(os.path.join(args.output, args.dataset, args.modals, args.name+'.pth')))
    with torch.no_grad():
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, test_loader, args=args)

    print('Test performance..')
    print('Loss {} accuracy {}'.format(test_loss, test_acc))
    print(classification_report(test_label, test_pred, target_names=target_names, sample_weight=test_mask, digits=4))
    print(confusion_matrix(test_label, test_pred, sample_weight=test_mask))

