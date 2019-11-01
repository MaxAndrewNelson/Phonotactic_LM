import torch
import torch.nn as nn

# single-direction RNN, optionally tied embeddings
class Emb_RNNLM(nn.Module):
    def __init__(self, params):
        super(Emb_RNNLM, self).__init__()
        self.vocab_size = params['inv_size']
        self.d_emb = params['d_emb']
        self.n_layers = params['num_layers']
        self.d_hid = params['d_hid']
        self.embeddings = nn.Embedding(self.vocab_size, self.d_emb)
        
        # input to recurrent layer, default nonlinearity is tanh
        self.i2R = nn.RNN(
            self.d_emb, self.d_hid, batch_first=True, num_layers = self.n_layers
        )
        # recurrent to output layer
        self.R2o = nn.Linear(self.d_hid, self.vocab_size)
        if params['tied']:
            if self.d_emb == self.d_hid:
                self.R2o.weight = self.embeddings.weight
            else:
                print("Dimensions don't support tied embeddings")

    def forward(self, batch):
        batches, seq_len = batch.size()
        embs = self.embeddings(batch)
        output, hidden = self.i2R(embs)
        outputs = self.R2o(output)
        return outputs

#single direction RNN with fixed, prespecified features
class Feature_RNNLM(nn.Module):
    def __init__(self, params, feature_table): #the ith row of the feature table is the set of features for word i in phone2ix
        super(Feature_RNNLM, self).__init__()
        self.features = feature_table
        self.vocab_size = params['inv_size']
        self.d_feats = params['d_feats']
        self.n_layers = params['num_layers']
        self.d_hid = params['d_hid']
        self.i2R = nn.RNN(self.d_feats, self.d_hid, batch_first=True, num_layers=self.n_layers) #input to recurrent layer, default nonlinearity is tanh
        self.R2o = nn.Linear(self.d_hid, self.vocab_size) #recurrent to output layer

    def batch_to_features(self, batch, feature_table):
        batches, seq_len = batch.size()

    def forward(self, batch):
        batches, seq_len = batch.size()
        inventory_size, num_feats = self.features.size()
        full_representation = torch.zeros(batches, seq_len, num_feats, requires_grad=False)#the final will be batch size x seq_len x number of features
        for i in range(batches):
            for j in range(seq_len):
                full_representation[i,j,:] = self.features[batch[i,j]]
        output, hidden = self.i2R(full_representation)
        outputs = self.R2o(output)
        return(outputs)