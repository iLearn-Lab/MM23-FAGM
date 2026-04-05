import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import get_IEMOCAP_loaders, get_MELD_loaders
from model import DialogueGNNModel
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from loss import FocalLoss
import os

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

def modulation_init(model, dataloader, cuda_flag, args):
    model.eval()
    with torch.no_grad():
        handles = []
        for name, module in model.named_modules():
            if not list(module.children()) and 'smax_fc' not in name:
                if list(module.parameters()) and not isinstance(module, torch.nn.Embedding):
                    handles.append(module.register_forward_hook(hook))

        data = next(iter(dataloader))
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        
        model(textf, qmask, lengths, acouf, visuf)
        all_act = get_act()

        if "v" in args.modals:
            model(torch.zeros_like(textf), qmask, lengths, torch.zeros_like(acouf), visuf)
            v_act = get_act()

        if "a" in args.modals:
            model(torch.zeros_like(textf), qmask, lengths, acouf, torch.zeros_like(visuf))
            a_act = get_act()

        if "l" in args.modals:
            model(textf, qmask, lengths, torch.zeros_like(acouf), torch.zeros_like(visuf))
            l_act = get_act()

        for key in all_act.keys():
            if "l" in args.modals:
                if torch.sum(all_act[key]-l_act[key]) == 0:
                    MODAL_SPEC['text'].append(key)
            if "a" in args.modals:
                if torch.sum(all_act[key]-a_act[key]) == 0:
                    MODAL_SPEC['audio'].append(key)
            if "v" in args.modals:
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

def modulation(model, all_prob, textf, qmask, lengths, acouf, visuf, label, step, args):
    model.eval()
    if step % args.tau == 0:
        with torch.no_grad():
            all_act = get_act()
            loss_all = loss_f(all_prob, label)

            if "l" in args.modals:
                el_prob = model(torch.zeros_like(textf), qmask, lengths, acouf, visuf)
                el_act = get_act()
                loss_el = loss_f(el_prob, label)

            if "v" in args.modals:
                ev_prob = model(textf, qmask, lengths, acouf, torch.zeros_like(visuf))
                ev_act = get_act()
                loss_ev = loss_f(ev_prob, label)

            if "a" in args.modals:
                ea_prob = model(textf, qmask, lengths, torch.zeros_like(acouf), visuf)
                ea_act = get_act()
                loss_ea = loss_f(ea_prob, label)

            if args.modals == "avl":
                score = torch.tensor([loss_ea-loss_all, loss_ev-loss_all, loss_el-loss_all])
            elif args.modals == "av":
                score = torch.tensor([loss_ea-loss_all, loss_ev-loss_all])
            elif args.modals == "al":
                score = torch.tensor([loss_ea-loss_all, loss_el-loss_all])
            elif args.modals == "vl":
                score = torch.tensor([loss_ev-loss_all, loss_el-loss_all])
            
            ratio = F.softmax(0.1*score, dim=0)
            r_min, _ = torch.min(ratio, dim=0)
            iscore = (ratio - r_min)**args.gamma

            if args.modals == "avl":
                iscore_a, iscore_v, iscore_l = iscore[0], iscore[1], iscore[2]
            elif args.modals == "av":
                iscore_a, iscore_v = iscore[0], iscore[1]
            elif args.modals == "al":
                iscore_a, iscore_l = iscore[0], iscore[1]
            elif args.modals == "vl":
                iscore_v, iscore_l = iscore[0], iscore[1]
            
            if "a" in args.modals:
                ISCORE.update({"audio": iscore_a})
            if "v" in args.modals:
                ISCORE.update({"video": iscore_v})
            if "l" in args.modals:
                ISCORE.update({"text": iscore_l})

    if "a" in args.modals:
        for module in MODAL_SPEC['audio']:
            for param in module.parameters():
                if param.grad is not None:
                    param.grad *= (1 - ISCORE["audio"])

    if "v" in args.modals:
        for module in MODAL_SPEC['visual']:
            for param in module.parameters():
                if param.grad is not None:
                    param.grad *= (1 - ISCORE["video"])
    
    if "l" in args.modals:
        for module in MODAL_SPEC['text']:
            for param in module.parameters():
                if param.grad is not None:
                    param.grad *= (1 - ISCORE["text"])
    
    for module in MODAL_GEN:
        if step % args.tau == 0:
            if "l" in args.modals:
                delta_l = torch.abs(all_act[module] - el_act[module])
                delta_l = torch.mean(delta_l.reshape(-1, delta_l.size(-1)), dim=0)
            else:
                delta_l = torch.zeros(all_act[module].size(-1)).cuda()

            if "a" in args.modals:
                delta_a = torch.abs(all_act[module] - ea_act[module])
                delta_a = torch.mean(delta_a.reshape(-1, delta_a.size(-1)), dim=0)
            else:
                delta_a = torch.zeros(all_act[module].size(-1)).cuda()
            
            if "v" in args.modals:
                delta_v = torch.abs(all_act[module] - ev_act[module])
                delta_v = torch.mean(delta_v.reshape(-1, delta_v.size(-1)), dim=0)
            else:
                delta_v = torch.zeros(all_act[module].size(-1)).cuda()

            # rate = (epoch/args.epochs)**args.beta
            rate = args.beta
            pen = torch.zeros(all_act[module].size(-1)).cuda()

            if "a" in args.modals:
                delta = delta_a - torch.max(torch.stack([delta_l, delta_v], dim=0), dim=0)[0]
                pen_a = get_penalty(delta, rate)
                pen += pen_a * ISCORE["audio"]
            
            if "v" in args.modals:
                delta = delta_v - torch.max(torch.stack([delta_a, delta_l], dim=0), dim=0)[0]
                pen_v = get_penalty(delta, rate)
                pen += pen_v * ISCORE["video"]

            if "l" in args.modals:
                delta = delta_l - torch.max(torch.stack([delta_a, delta_v], dim=0), dim=0)[0]
                pen_l = get_penalty(delta, rate)
                pen += pen_l * ISCORE["text"]

            pen = 1-pen
            PEN.update({module: pen})

        for param in module.parameters():
            if param.grad is not None:
                if len(param.grad.size()) > 1: # weight
                    if module.__class__.__name__ in ['LSTM', 'GRU'] and param.grad.size()[1] != PEN[module].size():
                        continue
                    
                    param.grad *= PEN[module].unsqueeze(0)
                   
    model.train()



def train_or_eval_graph_model(model, loss_f, dataloader, epoch=0, train_flag=False, optimizer=None, cuda_flag=False, args=None,
                              test_label=False):
    losses, preds, labels = [], [], []

    assert not train_flag or optimizer != None
    if train_flag:
        model.train()
    else:
        model.eval()

    step = 0
    for data in dataloader:
        if train_flag:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        if args.multi_modal:
            if args.mm_fusion_mthd == 'concat':
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf], dim=-1)
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf], dim=-1)
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf], dim=-1)
                elif modals == 'al':
                    textf = torch.cat([acouf, textf], dim=-1)
                else:
                    raise NotImplementedError
            elif args.mm_fusion_mthd == 'gated':
                textf = textf
        else:
            if modals == 'a':
                textf = acouf
            elif modals == 'v':
                textf = visuf
            elif modals == 'l':
                textf = textf
            else:
                raise NotImplementedError

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        handles = []
        if train_flag == True and args.modulation == True and step % args.tau == 0:
            for module in MODAL_GEN:
                handles.append(module.register_forward_hook(hook))
        
        if args.multi_modal and args.mm_fusion_mthd in ['gated', 'mfn', 'mfn_only', 'concat_subsequently', 'tfn_only', 'lmf_only', 'concat_only']:
            log_prob = model(textf, qmask,  lengths, acouf, visuf)
        else:
            log_prob = model(textf, qmask, lengths)

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_f(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train_flag == True:
            loss.backward()
            
            if args.modulation == True:

                modulation(model, log_prob, textf, qmask, lengths, acouf, visuf, label, step, args)
                for handle in handles:
                    handle.remove()

            optimizer.step()
            step += 1

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return [], [], float('nan'), float('nan'), [], [], float('nan'), []

    labels = np.array(labels)
    preds = np.array(preds)
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--dataset', default='MELD', help='dataset to train and test')
    parser.add_argument('--data_dir', type=str, default='./MM-DFN/data/meld/MELD_features_raw1.pkl', help='dataset dir')
    parser.add_argument('--multi_modal', action='store_true', default=False, help='whether to use multimodal information')
    parser.add_argument('--modals', default='avl', help='modals to fusion: avl')
    parser.add_argument('--mm_fusion_mthd', default='concat_subsequently',
                        help='method to use multimodal information: mfn, concat, gated, concat_subsequently,mfn_only,tfn_only,lmf_only')
    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')
    parser.add_argument('--use_residue', action='store_true', default=False, help='whether to use residue information or not')
    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality')
    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')
    parser.add_argument('--use_crn_speaker', action='store_true', default=True, help='whether to use use crn_speaker embedding')
    parser.add_argument('--speaker_weights', type=str, default='3-0-1', help='speaker weight 0-0-0')
    parser.add_argument('--use_speaker', action='store_true', default=False, help='whether to use speaker embedding')
    parser.add_argument('--reason_flag', action='store_true', default=True, help='reason flag')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--valid_rate', type=float, default=0.1, metavar='valid_rate', help='valid rate, 0.0/0.1')
    parser.add_argument('--modal_weight', type=float, default=1.0, help='modal weight 1/0.7')
    parser.add_argument('--Deep_GCN_nlayers', type=int, default=16, help='Deep_GCN_nlayers')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha 0.1/0.2')
    parser.add_argument('--lamda', type=float, default=0.5, help='eta 0.5/0')
    parser.add_argument('--focal', type=float, default=0.5, help='focal 0.5/1/2')
    parser.add_argument('--loss', default="FocalLoss", help='loss function: FocalLoss/NLLLoss')
    parser.add_argument('--class_weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--output', type=str, default='./outputs', help='saved model dir')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--test_label', action='store_true', default=False, help='whether do test only')
    parser.add_argument('--load_model', type=str, default='./outputs', help='trained model dir')
    parser.add_argument('--name', type=str, default='demo', help='Experiment name')
    parser.add_argument('--log_dir', type=str, default='log/', help='tensorboard save path')
    parser.add_argument('--beta', type=float, default=1, help='')
    parser.add_argument('--gamma', type=float, default=1, help='')
    parser.add_argument('--tau', type=float, default=1, help='')
    parser.add_argument('--modulation', action='store_true', default=False, help='Enables grad modulation')
    args = parser.parse_args()

    print(args)

    cuda_flag = torch.cuda.is_available() and not args.no_cuda

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    n_epochs = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = feat2dim['textCNN'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        if args.mm_fusion_mthd == 'concat':
            if modals == 'avl':
                D_m = D_audio + D_visual + D_text
            elif modals == 'av':
                D_m = D_audio + D_visual
            elif modals == 'al':
                D_m = D_audio + D_text
            elif modals == 'vl':
                D_m = D_visual + D_text
            else:
                raise NotImplementedError
        else:
            D_m = D_text
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError

    D_e = 100
    graph_h = 100

    n_speakers, n_classes, class_weights, target_names = -1, -1, None, None
    if args.dataset == 'IEMOCAP':
        n_speakers, n_classes = 2, 6
        target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
        class_weights = torch.FloatTensor([1 / 0.086747,
                                           1 / 0.144406,
                                           1 / 0.227883,
                                           1 / 0.160585,
                                           1 / 0.127711,
                                           1 / 0.252668])
    if args.dataset == 'MELD':
        n_speakers, n_classes = 9, 7
        target_names = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
        class_weights = torch.FloatTensor([1.0 / 0.466750766,
                                           1.0 / 0.122094071,
                                           1.0 / 0.027752748,
                                           1.0 / 0.071544422,
                                           1.0 / 0.171742656,
                                           1.0 / 0.026401153,
                                           1.0 / 0.113714183])

    model = DialogueGNNModel(   D_m, D_e, graph_h,
                                n_speakers=n_speakers,
                                n_classes=n_classes,
                                dropout=args.dropout,
                                no_cuda=not cuda_flag,
                                alpha=args.alpha,
                                lamda=args.lamda,
                                use_residue=args.use_residue,
                                D_m_v=D_visual,
                                D_m_a=D_audio,
                                modals=args.modals,
                                att_type=args.mm_fusion_mthd,
                                av_using_lstm=args.av_using_lstm,
                                Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                dataset=args.dataset,
                                use_speaker=args.use_speaker,
                                use_modal=args.use_modal,
                                reason_flag=args.reason_flag,
                                multi_modal=args.multi_modal,
                                use_crn_speaker=args.use_crn_speaker,
                                speaker_weights=args.speaker_weights,
                                modal_weight=args.modal_weight,
                                )

    
    print('Running on the {} features........'.format(modals))

    if cuda_flag:
        print('Running on GPU')
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')

    if args.loss == 'FocalLoss':
        loss_f = FocalLoss(gamma=args.focal, alpha=class_weights if args.class_weight else None)
    else:
        loss_f = nn.NLLLoss(class_weights if args.class_weight else None)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(data_path=args.data_dir,
                                                                   valid_rate=args.valid_rate,
                                                                   batch_size=batch_size,
                                                                   num_workers=0)
    elif args.dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(data_path=args.data_dir,
                                                                      valid_rate=args.valid_rate,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        train_loader, valid_loader, test_loader = None, None, None
        print("There is no such dataset")

    if args.test_label == False:
        best_fscore = None
        counter = 0

        if args.modulation:
            modulation_init(model, train_loader, cuda_flag, args)

        for e in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc, _, _, train_fscore = train_or_eval_graph_model(model=model,
                                                                                loss_f=loss_f,
                                                                                dataloader=train_loader,
                                                                                epoch=e,
                                                                                train_flag=True,
                                                                                optimizer=optimizer,
                                                                                cuda_flag=cuda_flag,
                                                                                args=args)
            end_time = time.time()
            train_time = round(end_time-start_time, 2)

            start_time = time.time()
            with torch.no_grad():
                valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_graph_model(model=model,
                                                                                    loss_f=loss_f,
                                                                                    dataloader=valid_loader,
                                                                                    epoch=e,
                                                                                    cuda_flag=cuda_flag,
                                                                                    args=args)
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
                    format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, train_time, valid_time))
            
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
        

    if args.test_label == True:
        model.load_state_dict(torch.load(args.load_model))
    else:
        model.load_state_dict(torch.load(os.path.join(args.output, args.dataset, args.modals, args.name+'.pth')))
    with torch.no_grad():
        test_loss, test_acc, test_label, test_pred, test_fscore = train_or_eval_graph_model(model=model,
                                                                                            loss_f=loss_f,
                                                                                            dataloader=test_loader,
                                                                                            train_flag=False,
                                                                                            cuda_flag=cuda_flag,
                                                                                            args=args,
                                                                                            test_label=True)
    print('Test performance..')
    print('Loss {}, accuracy {}'.format(test_loss, test_acc))
    print(classification_report(test_label, test_pred, digits=4))
    print(confusion_matrix(test_label, test_pred))