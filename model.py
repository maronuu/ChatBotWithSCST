import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

import utils

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
HIDDEN_SIZE = 512
EMBEDDING_DIM = 50


class Seq2seqModel(nn.Module):
    def __init__(self, embedding_dim, hidden_size, dict_size):
        super(Seq2seqModel, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=dict_size,
            embedding_dim=embedding_dim,
        )
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, dict_size)

    def encode(self, x):
        _, hidden = self.encoder(x)
        return hidden
    
    
    def decode_train_data(self, hidden, input_seq):
        out, _ = self.decoder(input_seq, hidden)
        out = self.output_layer(out.data)

        return out
    
    def decode_token(self, hidden, input_token):
        out, updated_hidden = self.decoder(input_token.unsqueeze(0), hidden)
        out = self.output_layer(out).squeeze(dim=0)
        
        return (out, updated_hidden)
    
    def decode_sequences(self, hidden, length, init_emb, mode, end_of_decoding=None):
        action_list = []
        logit_list = []

        current_emb = init_emb
        for _ in range(length):
            logit, hidden = self.decode_token(hidden, current_emb)
            if mode == 'sampling':
                probs_tensor = F.softmax(logit, dim=1)
                probs = probs_tensor.data.cpu().numpy()[0]
                action = int(np.random.choice(
                    probs.shape[0], p=probs
                ))
                action_tensor = torch.LongTensor([action])
                action_tensor = action_tensor.to(init_emb.device)
                
            elif mode == 'argmax':
                action_tensor = torch.max(logit, dim=1)[1]
                action = action_tensor.data.cpu().numpy()[0]

            current_emb = self.embedding(action_tensor)

            action_list.append(action)
            logit_list.append(logit)

            if end_of_decoding is not None and \
                action == end_of_decoding:
                break
        
        return (torch.cat(logit_list), action_list)

    def get_encoded_item(self, encoded, index):
        item0 = encoded[0][:, index:index+1].contiguous()
        item1 = encoded[1][:, index:index+1].contiguous()
        return (item0, item1)