import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import get_IEMOCAP_loaders, get_MELD_loaders
from model import MultiDialogueGCN
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import os

def get_penalty(tensor, rate):
    pos_num = torch.sum((tensor>0).int()).item()
    sorted = torch.sort(tensor, descending=True)[0]
    threshold = sorted[max(int(pos_num*rate)-1, 0)]
    penalty = (tensor > threshold).float()
    return penalty

ISCORE = {}

def modulation(model, all_prob, hiddens, label, step, args):
    if step % args.tau == 0:
        with torch.no_grad():
            loss_all = loss_f(F.log_softmax(all_prob, 1), label)

            if args.modals == "avl":
                ea_prob = model.smax_fc(torch.cat([torch.zeros_like(hiddens[0]), hiddens[1], hiddens[2]], dim=-1))
                loss_ea = loss_f(F.log_softmax(ea_prob, 1), label)
                ev_prob = model.smax_fc(torch.cat([hiddens[0], torch.zeros_like(hiddens[1]), hiddens[2]], dim=-1))
                loss_ev = loss_f(F.log_softmax(ev_prob, 1), label)
                el_prob = model.smax_fc(torch.cat([hiddens[0], hiddens[1], torch.zeros_like(hiddens[2])], dim=-1))
                loss_el = loss_f(F.log_softmax(el_prob, 1), label)
                score = torch.tensor([loss_ea-loss_all, loss_ev-loss_all, loss_el-loss_all])
            elif args.modals in ["av", "al", "vl"]:
                e0_prob = model.smax_fc(torch.cat([torch.zeros_like(hiddens[0]), hiddens[1]], dim=-1))
                loss_e0 = loss_f(F.log_softmax(e0_prob, 1), label)
                e1_prob = model.smax_fc(torch.cat([hiddens[0], torch.zeros_like(hiddens[1])], dim=-1))
                loss_e1 = loss_f(F.log_softmax(e1_prob, 1), label)
                score = torch.tensor([loss_e0-loss_all, loss_e1-loss_all])

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
        for param in model.dialog_a.parameters():
            if param.grad is not None:
                param.grad *= (1 - ISCORE["audio"])
    
    if "v" in args.modals:
        for param in model.dialog_v.parameters():
            if param.grad is not None:
                param.grad *= (1 - ISCORE["video"])

    if "l" in args.modals:
        for param in model.dialog_l.parameters():
            if param.grad is not None:
                param.grad *= (1 - ISCORE["text"])


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
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
       
        log_prob, hiddens = model(textf, qmask, umask, lengths, acouf, visuf)

        pred = F.log_softmax(log_prob, 1)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_f(pred, label)
        preds.append(torch.argmax(pred, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train_flag == True:
            loss.backward()

            if args.modulation == True:
                modulation(model, log_prob, hiddens, label, step, args)

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
    parser.add_argument('--data_dir', type=str, default='./data/meld/MELD_features_raw1.pkl', help='dataset dir')
    parser.add_argument('--modals', default='avl', help='modals to fusion: avl')
    parser.add_argument('--windowp', type=int, default=2, help='context window size for constructing edges in graph model for past utterances')
    parser.add_argument('--windowf', type=int, default=2, help='context window size for constructing edges in graph model for future utterances')
    parser.add_argument('--nodal-attention', action='store_true', default=False, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--valid_rate', type=float, default=0.1, metavar='valid_rate', help='valid rate, 0.0/0.1')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--class_weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--output', type=str, default='./outputs', help='saved model dir')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--test_label', action='store_true', default=False, help='whether do test only')
    parser.add_argument('--test_modal', default='l', help='whether do test only')
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
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = feat2dim['textCNN'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_text']


    D_e = 100
    D_m = 100
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

    model = MultiDialogueGCN( args.modals, D_audio, D_visual, D_text, D_m, D_e, graph_h, n_speakers,
                                 max_seq_len=110,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda)
    
    print('Running on the {} features........'.format(args.modals))

    if cuda_flag:
        print('Running on GPU')
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')


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