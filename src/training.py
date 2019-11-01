import random
import numpy as np
import torch
import torch.nn as nn
import time

def compute_perplexity(dataset, net, bsz=64):
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    num_examples, seq_len = dataset.size()
    
    batches = [(start, start + bsz) for start in\
               range(0, num_examples, bsz)]
    
    total_unmasked_tokens = 0.
    nll = 0.
    for b_idx, (start, end) in enumerate(batches):
            
        batch = dataset[start:end]
        ut = torch.nonzero(batch).size(0)
        preds = net(batch)
        targets = batch[:, 1:].contiguous().view(-1)
        preds = preds[:, :-1, :].contiguous().view(-1, net.vocab_size)
        loss = criterion(preds, targets)
        nll += loss.detach()
        total_unmasked_tokens += ut

    perplexity = torch.exp(nll / total_unmasked_tokens).cpu()
    return perplexity.data
    
def train_lm(dataset, dev, params, net):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(net.parameters(), lr=params['learning_rate'])
    num_examples, seq_len = dataset.size()    
    batches = [
        (start, start + params['batch_size']) 
        for start in range(0, num_examples, params['batch_size'])
    ]
    
    prev_perplexity = 1e10
    for epoch in range(params['epochs']):
        ep_loss = 0.
        start_time = time.time()
        random.shuffle(batches)
        
        for b_idx, (start, end) in enumerate(batches):
            batch = dataset[start:end]
            preds = net(batch)
            preds = preds[:, :-1, :].contiguous().view(-1, net.vocab_size)
            targets = batch[:, 1:].contiguous().view(-1)
            loss = criterion(preds, targets)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ep_loss += loss.detach() 
        dev_perplexity = compute_perplexity(dev,net)

        print('epoch: %d, loss: %0.2f, time: %0.2f sec, dev perplexity: %0.2f' %
              (epoch, ep_loss, time.time()-start_time, dev_perplexity))
        # stop early criterion, increasing perplexity on dev 
        if dev_perplexity - prev_perplexity > 0.01:
            print('Stop early reached')
            break

