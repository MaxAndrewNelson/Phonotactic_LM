import torch
import numpy as np
from scipy.stats import norm
from statsmodels.stats.weightstats import CompareMeans
from training import compute_perplexity

def get_probs(input_file, model, phone2ix, out_filename):
    inp_file = open(input_file, 'r',encoding='UTF-8')
    out_file = open(out_filename,'w',encoding='UTF-8')
    data_tens = []
    as_strings = []
    for line in inp_file:
        line = line.rstrip()
        as_strings.append(line.replace(' ',''))
        line = line.split(' ')
        line = ['<s>'] + line + ['<e>']
        line_as_tensor = torch.LongTensor([phone2ix[p] for p in line])
        data_tens.append(line_as_tensor)

    num_points = len(data_tens)

    for i,word in enumerate(data_tens):
        curr_string = as_strings[i]
        out_file.write(curr_string + '\t' + str(compute_perplexity(word.unsqueeze(0), model).numpy()) + '\n')
    
    inp_file.close()
    out_file.close()