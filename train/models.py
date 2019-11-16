import torch
import torch.nn as nn
from preprocessing.nn_dataset import MAX_SEQ
from preprocessing.nn_dataset import position_encoding_init
from train.external import VariationalDropout
from train.transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

"""
File containing classes representing various Neural architectures
"""


# MARK:- TonicNet
class TonicNet(nn.Module):
    def __init__(self, nb_tags, nb_layers=1, z_dim =0,
                 nb_rnn_units=100, batch_size=1, seq_len=1, dropout=0.0):
        super(TonicNet, self).__init__()

        self.nb_layers = nb_layers
        self.nb_rnn_units = nb_rnn_units
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dropout = dropout
        self.z_dim = z_dim
        self.z_emb_size = 32

        self.nb_tags = nb_tags

        # build actual NN
        self.__build_model()

    def __build_model(self):

        self.embedding = nn.Embedding(self.nb_tags, self.nb_rnn_units)

        # Unused but key exists in state_dict
        self.pos_emb = nn.Embedding(64, 0)

        self.z_embedding = nn.Embedding(80, self.z_emb_size)

        self.dropout_i = VariationalDropout(max(0.0, self.dropout - 0.2), batch_first=True)

        # design RNN

        input_size = self.nb_rnn_units
        if self.z_dim > 0:
            input_size += self.z_emb_size

        self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=self.nb_rnn_units,
                num_layers=self.nb_layers,
                batch_first=True,
                dropout=self.dropout
            )
        self.dropout_o = VariationalDropout(self.dropout, batch_first=True)

        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(input_size, self.nb_tags, bias=False)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_rnn_units)
        hidden_a = torch.randn(self.nb_layers, self.batch_size, self.nb_rnn_units)

        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()

        return hidden_a

    def forward(self, X, z=None, train_embedding=True, sampling=False, reset_hidden=True):
        # reset the RNN hidden state.
        if not sampling:
            self.seq_len = X.shape[1]
            if reset_hidden:
                self.hidden = self.init_hidden()

        self.embedding.weight.requires_grad = train_embedding

        # ---------------------
        # Combine inputs
        X = self.embedding(X)
        X = X.view(self.batch_size, self.seq_len, self.nb_rnn_units)

        # repeating pitch encoding
        if self.z_dim > 0:
            Z = self.z_embedding(z % 80)
            Z = Z.view(self.batch_size, self.seq_len, self.z_emb_size)
            X = torch.cat((Z, X), 2)

        X = self.dropout_i(X)

        # Run through RNN
        X, self.hidden = self.rnn(X, self.hidden)

        if self.z_dim > 0:
            X = torch.cat((Z, X), 2)

        X = self.dropout_o(X)

        # run through linear layer
        X = self.hidden_to_tag(X)

        Y_hat = X
        return Y_hat


class Transformer_Model(nn.Module):
    def __init__(self, nb_tags, nb_layers=1, pe_dim=0,
                 emb_dim=100, batch_size=1, seq_len=MAX_SEQ, dropout=0.0, encoder_only=True):
        super(Transformer_Model, self).__init__()

        self.nb_layers = nb_layers
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pe_dim = pe_dim
        self.dropout = dropout

        self.nb_tags = nb_tags

        self.encoder_only = encoder_only

        # build actual NN
        self.__build_model()

    def __build_model(self):

        self.embedding = nn.Embedding(self.nb_tags, self.emb_dim)

        if not self.encoder_only:
            self.embedding2 = nn.Embedding(self.nb_tags, self.emb_dim)

        self.pos_emb = position_encoding_init(MAX_SEQ, self.pe_dim)
        self.pos_emb.requires_grad = False

        self.dropout_i = nn.Dropout(self.dropout)

        input_size = self.pe_dim + self.emb_dim

        self.transformerLayerI = TransformerEncoderLayer(d_model=input_size,
                                                         nhead=8,
                                                         dropout=self.dropout,
                                                         dim_feedforward=1024)

        self.transformerI = TransformerEncoder(self.transformerLayerI,
                                               num_layers=self.nb_layers,)

        self.dropout_m = nn.Dropout(self.dropout)

        if not self.encoder_only:
            # design decoder
            self.transformerLayerO = TransformerDecoderLayer(d_model=input_size,
                                                             nhead=8,
                                                             dropout=self.dropout,
                                                             dim_feedforward=1024)

            self.transformerO = TransformerDecoder(self.transformerLayerO,
                                                   num_layers=self.nb_layers, )

            self.dropout_o = nn.Dropout(self.dropout)

        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(self.emb_dim + self.pe_dim, self.nb_tags)

    def __pos_encode(self, p):
        return self.pos_emb[p]

    def forward(self, X, p, X2=None, train_embedding=True):

        self.embedding.weight.requires_grad = train_embedding
        if not self.encoder_only:
            self.embedding2.weight.requires_grad = train_embedding

        I = X

        self.mask = (torch.triu(torch.ones(self.seq_len, self.seq_len)) == 1).transpose(0, 1)
        self.mask = self.mask.float().masked_fill(self.mask == 0, float('-inf')).masked_fill(self.mask == 1, float(0.0))

        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

        # ---------------------
        # Combine inputs
        X = self.embedding(I)
        X = X.view(self.seq_len, self.batch_size, -1)

        if self.pe_dim > 0:
            P = self.__pos_encode(p)
            P = P.view(self.seq_len, self.batch_size, -1)
            X = torch.cat((X, P), 2)

        X = self.dropout_i(X)

        # Run through transformer encoder

        M = self.transformerI(X, mask=self.mask)
        M = self.dropout_m(M)

        if not self.encoder_only:
            # ---------------------
            # Decoder stack
            X = self.embedding2(X2)
            X = X.view(self.seq_len, self.batch_size, -1)

            if self.pe_dim > 0:
                X = torch.cat((X, P), 2)

            X = self.dropout_i(X)

            X = self.transformerO(X, M, tgt_mask=self.mask, memory_mask=None)
            X = self.dropout_o(X)

            # run through linear layer
            X = self.hidden_to_tag(X)
        else:
            X = self.hidden_to_tag(M)

        Y_hat = X
        return Y_hat


# MARK:- Custom s2s Cross Entropy loss
class CrossEntropyTimeDistributedLoss(nn.Module):
    """loss function for multi-timsetep model output"""
    def __init__(self):
        super(CrossEntropyTimeDistributedLoss, self).__init__()

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, y_hat, y):

        _y_hat = y_hat.squeeze(0)
        _y = y.squeeze(0)

        # Loss from one sequence
        loss = self.loss_func(_y_hat, _y)
        loss = torch.sum(loss)
        return loss


