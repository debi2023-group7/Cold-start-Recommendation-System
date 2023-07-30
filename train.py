# from metrics import *
import torch
from sklearn.metrics import log_loss, roc_auc_score
import time
# from trainer import *
import torch.nn as nn
import torch.nn.utils.prune as prune
from copy import deepcopy
import pandas
import matplotlib.pyplot as plt


def recalls_and_ndcgs_for_ks(scores, labels, ks, args):
    metrics = {}

    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float()).to(args.device)
       dcg = (hits * weights).sum(1)
       idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count]).to(args.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg

    return metrics
####################################################################################################################

def train(model, dataloader, val_loader,  writer, args): 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(args.device)

    best_metric = 0
    all_time = 0
    val_all_time = 0
    
        
    for epoch in range(args.epochs):
        since = time.time()
        ####################################################
        # Train
        model.train()
        running_loss = 0
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        correct = 0
        total = 0
        
        for data in dataloader:
            optimizer.zero_grad()
            data = [x.to(args.device) for x in data]
            seqs, labels = data
            logits = model(seqs)
            if 'cold' in args.task_name:
                logits = logits.mean(1)
                labels = labels.view(-1)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().item()
        
        print("Training CE Loss: {:.5f}".format(running_loss / len(dataloader)))
##################################################################
        tmp = time.time() - since
        print('one epoch train time:', tmp)
        all_time += tmp
    
        # Val
        val_since = time.time()
#**********************************************************************
        print("+" * 20, "Valid Epoch {}".format(epoch + 1), "+" * 20)
        model.eval()
        metrics = {}
        i = 0; test = False
        
        with torch.no_grad():
            tqdm_dataloader = val_loader
            for data in tqdm_dataloader:
                data = [x.to(args.device) for x in data]
                seqs, labels = data
                if test:
                    scores = model.predict(seqs)
                else:
                    scores = model(seqs)
                scores = scores.mean(1)
                metrics = recalls_and_ndcgs_for_ks(scores, labels, args.metric_ks, args)
                i += 1
                for key, value in metrics.items():
                    if key not in metrics:
                        metrics[key] = value
                    else:
                        metrics[key] += value
        for key, value in metrics.items():
            metrics[key] = value / i
        #         print(metrics)
        for k in sorted(args.metric_ks, reverse=True):
            writer.add_scalar('Train/NDCG@{}'.format(k), metrics['NDCG@%d' % k], epoch)
            print('Train/NDCG@{}'.format(k), metrics['NDCG@%d' % k], epoch)


#####################################################################3
        val_tmp = time.time() - val_since
        print('one epoch val time:', val_tmp)
        val_all_time += val_tmp
        if args.is_pretrain == 0 and 'acc' in args.task_name:
            if metrics['NDCG@20'] >= 0.0193:
                break
        i = 1
        current_metric = metrics['NDCG@5']
        if best_metric <= current_metric:
            best_metric = current_metric
            best_model = deepcopy(model)
        else:
            i += 1
            if i == 10:
                print('early stop!')
                break
        tmp = time.time() - since
        print('one epoch train time:', tmp)
        all_time += tmp
    print('train_time:', all_time)
    print('val_time:', val_all_time)

    return best_model

#####################################################################################################################


import torch
import torch.nn as nn
import time

def trainV2(model, dataloader, val_loader, writer, args):
    result =  pandas.DataFrame()
    X=  []; Y =[]; Z= [];

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(args.device)

    best_metric = 0
    all_time = 0
    val_all_time = 0
    i = 1  # Counter for early stopping

    for epoch in range(args.epochs):
        since = time.time()
        ####################################################
        # Train
        model.train()
        running_loss = 0
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        correct = 0
        total = 0

        for data in dataloader:
            optimizer.zero_grad()
            data = [x.to(args.device) for x in data]
            seqs, labels = data
            logits = model(seqs)
            if 'cold' in args.task_name:
                logits = logits.mean(1)
                labels = labels.view(-1)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().item()

        avg_train_loss = running_loss / len(dataloader)
        print("Training CE Loss: {:.5f}".format(avg_train_loss))
        ##################################################################
        tmp = time.time() - since
        print('one epoch train time:', tmp)
        all_time += tmp

        # Val
        val_since = time.time()
        # **********************************************************************
        print("+" * 20, "Valid Epoch {}".format(epoch + 1), "+" * 20)
        model.eval()
        metrics = {}
        i = 0  # Counter for the number of batches in validation
        test = False

        with torch.no_grad():
            for data in val_loader:
                data = [x.to(args.device) for x in data]
                seqs, labels = data
                if test:
                    scores = model.predict(seqs)
                else:
                    scores = model(seqs)
                scores = scores.mean(1)
                metrics_batch = recalls_and_ndcgs_for_ks(scores, labels, args.metric_ks, args)
                i += 1
                for key, value in metrics_batch.items():
                    if key not in metrics:
                        metrics[key] = value
                    else:
                        metrics[key] += value

        for key in metrics.keys():
            new_data = {}
            metrics[key] /= i
            writer.add_scalar('Train/{}'.format(key), metrics[key], epoch)
            
            # Append the new data to the DataFrame
#             print({'metrics': key, 'result': metrics[key], 'epoch': epoch})
            X = key ; Y= metrics[key] if X not in ['NDCG@5','NDCG@20' ] else   metrics[key].cpu().item() ; Z=epoch;
#             print({'metrics': key, 'result': metrics[key], 'epoch': epoch})
            new_data = pandas.DataFrame({'metrics': [X], 'result': [Y], 'epoch': [Z]})
            result = pandas.concat([new_data, result])
            
            print('Train/{}'.format(key), metrics[key], epoch)

        #####################################################################3
        val_tmp = time.time() - val_since
        print('one epoch val time:', val_tmp)
        val_all_time += val_tmp

        current_metric = metrics['NDCG@5']
        if best_metric <= current_metric:
            best_metric = current_metric
            # Save the model state dict
            torch.save(model.state_dict(), 'best_model.pth')
            i = 1  # Reset the early stopping counter
        else:
            i += 1
            if i == 10:
                print('early stop!')
                break

        tmp = time.time() - since
        print('one epoch train time:', tmp)
        all_time += tmp

    print('train_time:', all_time)
    print('val_time:', val_all_time)
    
    
#     result = pandas.DataFrame({'metrics': [X], 'result': [Y], 'epoch': [Z]})
#     print(result)
#             result = pandas.concat([new_data, result])
    # Pivot the DataFrame to have metrics as columns and epoch as index
    pivot_df = result.pivot(index='epoch', columns='metrics', values='result')

    # Plot the data
    pivot_df.plot(marker='o', linestyle='-', markersize=8)

    # Set plot labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Result')
    plt.xticks(range(0, result['epoch'].max()+1))
    plt.title('Results of Three Metrics Over Epochs')

    # Show the plot
    plt.legend(title='Metrics')
    plt.show()

    del result,new_data,X,Y,Z
    return model