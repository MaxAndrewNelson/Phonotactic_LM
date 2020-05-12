import random
import torch

def get_corpus_data(filename):
    """
    Reads input file and coverts it to list of lists, adding word boundary 
    markers.
    """
    raw_data = []
    file = open(filename,'r')
    for line in file:
        line = line.rstrip()
        line = ['<s>'] + line.split(' ') + ['<e>']
        raw_data.append(line)
    return raw_data

def process_data(string_training_data, dev=True, training_split=60):
    random.shuffle(string_training_data)
    # all data points need to be padded to the maximum length
    max_chars = max([len(x) for x in string_training_data])
    string_training_data = [
        sequence + ['<p>'] * (max_chars - len(sequence)) 
        for sequence in string_training_data]
    # get the inventory and build both directions of dicts  
    # this will store the set of possible phones
    inventory = list(set(phone for word in string_training_data for phone in word))
    inventory = ['<p>'] + [x for x in inventory if x != '<p>'] #ensure that the padding symbol is at index 0

    # dictionaries for looking up the index of a phone and vice versa
    phone2ix = {p: ix for (ix, p) in enumerate(inventory)}
    ix2phone = {ix: p for (ix, p) in enumerate(inventory)}

    as_ixs = [
        torch.LongTensor([phone2ix[p] for p in sequence]) 
        for sequence in string_training_data
      ]

    if not dev:
        training_data = torch.stack(as_ixs, 0)
        # simpler make a meaningless tiny dev than to have a different eval 
        # training method that doesn't compute Dev perplexity
        dev = torch.stack(as_ixs[-10:], 0)
    else:
        split = int(len(as_ixs) * (training_split/100))
        training_data = torch.stack(as_ixs[:split], 0)
        dev = torch.stack(as_ixs[split:], 0)

    return inventory, phone2ix, ix2phone, training_data, dev

def process_features(file_path, inventory):
    feature_dict = {}

    file = open(file_path,'r')
    header = file.readline()

    for line in file:
        line = line.rstrip()
        line = line.split(',')

        if line[0] in inventory:
            feature_dict[line[0]] = [1. if x=='+' else 0. if x=='0' else -1. for x in line[1:]]
            feature_dict[line[0]] += [0.,0.,0.]

    num_feats = len(feature_dict[line[0]])

    feature_dict['<s>'] = [0. for x in range(num_feats-3)] + [-1.0,-1.0,1.0]
    feature_dict['<p>'] = [0. for x in range(num_feats-3)] + [-1.0,1.0,-1.0]
    feature_dict['<e>'] = [0. for x in range(num_feats-3)] + [1.0,-1.0,-1.0]

    return(feature_dict, num_feats)
