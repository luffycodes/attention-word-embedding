import numpy as np
import time, sys, random

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import math
import time

from torch import FloatTensor as FT
from torch import ByteTensor as BT

TINY = 1e-11

class Word2MatEncoder(nn.Module):

    def __init__(self, n_words, word_emb_dim = 784, padding_idx = 0, w2m_type = "cbow", initialization_strategy = "identity", ngram_to_index=None, wordId2ngramIndices=None):
        """
        TODO: Method description for w2m encoder.
        """
        super(Word2MatEncoder, self).__init__()
        self.word_emb_dim = word_emb_dim
        self.key_query_dim = 50
        self.word_ngram_idx_dim = 10
        self.n_words = n_words
        self.w2m_type = w2m_type
        self.initialization_strategy = initialization_strategy

        # check that the word embedding size is a square
        # assert word_emb_dim == int(math.sqrt(word_emb_dim)) ** 2

        # type of aggregation to use to combine two word matrices
        self.w2m_type = w2m_type
        if self.w2m_type not in ["cbow", "cmow", "acbow", "acbow_fasttext"]:
            raise NotImplementedError("Operator " + self.operator + " is not yet implemented.")

        # set initial weights of word embeddings depending on the initialization strategy
        ## set weights of padding symbol such that it is the neutral element with respect to the operation
        if self.w2m_type == "cmow":
            neutral_element = np.reshape(np.eye(int(np.sqrt(self.word_emb_dim)), dtype=np.float32), (1, -1))
            neutral_element = torch.from_numpy(neutral_element)
        elif self.w2m_type == "cbow":
            neutral_element = np.reshape(torch.from_numpy(np.zeros((self.word_emb_dim), dtype=np.float32)), (1, -1))
        elif self.w2m_type == "acbow":
            neutral_element = np.reshape(torch.from_numpy(np.zeros((self.word_emb_dim), dtype=np.float32)), (1, -1))
            neutral_element_key = np.reshape(torch.from_numpy(np.zeros((self.key_query_dim), dtype=np.float32)),
                                             (1, -1))
            neutral_element_query = np.reshape(torch.from_numpy(np.zeros((self.key_query_dim), dtype=np.float32)),
                                               (1, -1))
        elif self.w2m_type == "acbow_fasttext":
            self.ngram_to_index = ngram_to_index
            self.wordId2ngramIndices = wordId2ngramIndices
            neutral_element = np.reshape(torch.from_numpy(np.zeros((self.word_emb_dim), dtype=np.float32)), (1, -1))
            neutral_element_key = np.reshape(torch.from_numpy(np.zeros((self.key_query_dim), dtype=np.float32)),
                                             (1, -1))
            neutral_element_query = np.reshape(torch.from_numpy(np.zeros((self.key_query_dim), dtype=np.float32)),
                                               (1, -1))
            neutral_element_ngram = np.reshape(torch.from_numpy(np.zeros((self.word_emb_dim), dtype=np.float32)), (1, -1))
            neutral_element_word_ngram_idx = np.reshape(torch.from_numpy(np.zeros((self.word_ngram_idx_dim), dtype=np.float32)),
                                                        (1, -1))
            neutral_element_output = np.reshape(torch.from_numpy(np.zeros((self.word_emb_dim), dtype=np.float32)), (1, -1))


        ## set weights of rest others depending on the initialization strategy
        if self.w2m_type == "cmow":
            if self.initialization_strategy == "identity":
                init_weights = self._init_random_identity()

            elif self.initialization_strategy == "normalized":
                ### normalized initialization by (Glorot and Bengio, 2010)
                init_weights = torch.from_numpy(np.random.uniform(size = (self.n_words,
                                                                          self.word_emb_dim),
                                                                  low = -np.sqrt(6 / (2*self.word_emb_dim)),
                                                                  high = +np.sqrt(6 / (2*self.word_emb_dim))
                                                                  ).astype(np.float32)
                                                )
            elif self.initialization_strategy == "normal":
                ### normalized with N(0,0.1), which failed in study by Yessenalina
                init_weights = torch.from_numpy(np.random.normal(size = (self.n_words,
                                                                         self.word_emb_dim),
                                                                 loc = 0.0,
                                                                 scale = 0.1
                                                                 ).astype(np.float32)
                                                )
            else:
                raise NotImplementedError("Unknown initialization strategy " + self.initialization_strategy)
        elif self.w2m_type == "cbow":
            init_weights = self._init_normal()
        elif self.w2m_type == "acbow":
            init_weights = self._init_normal()

            # set up key & query matrices and initialize them
            init_weights_key = self._init_key_query_normal_()
            weights_key = torch.cat([neutral_element_key, init_weights_key], dim=0)
            self.key_table = nn.Embedding(self.n_words + 1,
                                          self.key_query_dim,
                                          padding_idx=padding_idx,
                                          sparse=False, _weight=weights_key)

            init_weights_query = self._init_key_query_normal_()
            weights_query = torch.cat([neutral_element_query, init_weights_query], dim=0)
            self.query_table = nn.Embedding(self.n_words + 1,
                                            self.key_query_dim,
                                            padding_idx=padding_idx,
                                            sparse=False, _weight=weights_query)

        elif self.w2m_type == "acbow_fasttext":
            # set up key & query matrices and initialize them

            init_weights_key = self._init_key_query_normal_()
            weights_key = torch.cat([neutral_element_key, init_weights_key], dim=0)
            self.key_table = nn.Embedding(self.n_words + 1,
                                          self.key_query_dim,
                                          padding_idx=padding_idx,
                                          sparse=False, _weight=weights_key)

            init_weights_query = self._init_key_query_normal_()
            weights_query = torch.cat([neutral_element_query, init_weights_query], dim=0)
            self.query_table = nn.Embedding(self.n_words + 1,
                                            self.key_query_dim,
                                            padding_idx=padding_idx,
                                            sparse=False, _weight=weights_query)

            # setting up ngram matrices
            # set up word to ngram idx matrix and initialize them
            #
            init_weights_word_ngram_idx = self._init_zeroes()
            weights_words_ngram_idx = torch.cat([neutral_element_word_ngram_idx, init_weights_word_ngram_idx], dim=0)
            for word_idx in wordId2ngramIndices:
                weights_words_ngram_idx[word_idx] = torch.Tensor(wordId2ngramIndices[word_idx])

            self.word_ngram_idx_table = nn.Embedding(self.n_words + 1,
                                                     self.word_ngram_idx_dim,
                                                     padding_idx=padding_idx,
                                                     sparse=False, _weight=weights_words_ngram_idx)
            self.word_ngram_idx_table.weight.requires_grad = False

            # set up ngram embedding matrix and initialize them
            # set up bigger lookup table and initialize them
            self.n_grams = len(ngram_to_index)
            init_weights = self._init_normal()
            init_weights_ngram = self._init_normal_par_(self.n_grams, self.word_emb_dim)

            weights = torch.cat([neutral_element,
                                 init_weights,
                                 neutral_element_ngram,
                                 init_weights_ngram],
                                dim=0)
            self.lookup_table = nn.Embedding(self.n_words + 1 + self.n_grams + 1,
                                             self.word_emb_dim,
                                             padding_idx=padding_idx,
                                             sparse = False, _weight=weights)

            # output embedding
            # init_weights_output = self._init_normal()
            # weights_output = torch.cat([neutral_element_output,
            #                      init_weights_output],
            #                     dim=0)
            # self.output_table = nn.Embedding(self.n_words + 1,
            #                                  self.word_emb_dim,
            #                                  padding_idx=padding_idx,
            #                                  sparse = False, _weight=weights_output)

        # concatenate and set weights in the lookup table
        # set up word embedding table
        if self.w2m_type != "acbow_fasttext":
            weights = torch.cat([neutral_element,
                                 init_weights],
                                dim=0)
            self.lookup_table = nn.Embedding(self.n_words + 1,
                                             self.word_emb_dim,
                                             padding_idx=padding_idx,
                                             sparse = False, _weight=weights)

    def forward(self, sent, masked_word=None):

        # if self.w2m_type != "acbow_fasttext":
        if self.w2m_type != "acbow_fasttext" or masked_word is None:
            # sent_emb = self.lookup_table(sent)
            word_ngram_indices = self.word_ngram_idx_table(sent).long()
            sent_emb = torch.sum(self.lookup_table(word_ngram_indices), dim=2)

            # mask_sum = (word_ngram_indices != 0).sum(dim=2).clamp(min=1)
            # mask_sum = mask_sum.unsqueeze_(-1)
            # sent_emb.div_(mask_sum)
        else:
            # word_ngram_indices = self.word_ngram_idx_table(sent)
            # sent_emb = torch.sum(self.lookup_table(sent), dim=2)
            word_ngram_indices = self.word_ngram_idx_table(sent.reshape((sent.shape[0], -1))).long()
            sent_emb = torch.sum(self.lookup_table(word_ngram_indices), dim=2)

            # mask_sum = (word_ngram_indices != 0).sum(dim=2).clamp(min=1)
            # mask_sum = mask_sum.unsqueeze_(-1)
            # sent_emb.div_(mask_sum)

        seq_length = sent_emb.size()[1]
        matrix_dim = self._matrix_dim()

        # aggregate matrices
        if self.w2m_type == "cmow":
            # reshape vectors to matrices
            word_matrices = sent_emb.view(-1, seq_length, matrix_dim, matrix_dim)
            cur_emb = self._continual_multiplication(word_matrices)
        elif self.w2m_type == "cbow":
            word_matrices = sent_emb.view(-1, seq_length, matrix_dim, matrix_dim)
            cur_emb = torch.sum(word_matrices, 1)
        elif self.w2m_type == "acbow" or self.w2m_type == "acbow_fasttext":
            if masked_word is None:
                # cbow_weights = torch.ones(sent.shape[0], sent.shape[1], 1).cuda()
                # cbow_weights[(sent == 0)] = 0
                # att_sent_emb = torch.mul(sent_emb, cbow_weights)
                # cur_emb = torch.sum(att_sent_emb, dim=1)
                att_sent_emb = torch.mul(sent_emb, (sent != 0).float().view(sent.shape[0], sent.shape[1], 1))
                cur_emb = torch.sum(att_sent_emb, dim=1)

                # ensemble code
                # sent_emb_uv = self.U(sent)
                # att_sent_emb_uv = torch.mul(sent_emb_uv,
                #                             (sent != 0).float().view(sent.shape[0], sent.shape[1], 1))
                # embedding_UV = torch.sum(att_sent_emb_uv, dim=1)
                # ensemble code ends
            else:
                query_cbow_sent_emb = self.query_table(sent[:,:,0])
                key_sent_emb = self.key_table(masked_word)
                key_sent_emb = key_sent_emb.view(-1, self.key_query_dim, 1)
                att_cbow_weights = torch.bmm(query_cbow_sent_emb, key_sent_emb)
                att_cbow_weights[(sent[:, :, 0] == 0)] = float('-100000')
                # att_cbow_weights = torch.softmax(att_cbow_weights, dim=1)
                att_cbow_weights = torch.exp(att_cbow_weights)
                att_sent_emb = torch.mul(sent_emb, att_cbow_weights)
                cur_emb = torch.sum(att_sent_emb, dim=1)

                # ensemble code
                # sent_emb_uv = self.U(sent)
                # att_sent_emb_uv = torch.mul(sent_emb_uv,
                #                             (sent != 0).float().view(sent.shape[0], sent.shape[1], 1))
                # embedding_UV = torch.sum(att_sent_emb_uv, dim=1)
                # ensemble code ends

        # flatten final matrix
        emb = self._flatten_matrix(cur_emb)

        return emb

    def _continual_multiplication(self, word_matrices):
        cur_emb = word_matrices[:, 0, :]
        for i in range(1, word_matrices.size()[1]):
            cur_emb = torch.bmm(cur_emb, word_matrices[:, i, :])
        return cur_emb

    def _flatten_matrix(self, m):
        return m.view(-1, self.word_emb_dim)

    def _unflatten_matrix(self, m):
        return m.view(-1, self._matrix_dim(), self._matrix_dim())

    def _matrix_dim(self):
        return int(np.sqrt(self.word_emb_dim))

    def _init_random_identity(self):
        """Random normal initialization around 0., but add 1. at the diagonal"""
        init_weights = np.random.normal(size = (self.n_words, self.word_emb_dim),
                                        loc = 0.,
                                        scale = 0.1
                                        ).astype(np.float32)
        for i in range(self.n_words):
            init_weights[i, :] += np.reshape(np.eye(int(np.sqrt(self.word_emb_dim)), dtype=np.float32), (-1,))
        init_weights = torch.from_numpy(init_weights)
        return init_weights

    def _init_zeroes(self):
        ### normal initialization around 0.
        init_weights = torch.from_numpy(np.zeros(shape = (self.n_words,
                                                          self.word_ngram_idx_dim)
                                                 , dtype = np.float32)
                                        )
        return init_weights

    def _init_normal(self):
        ### normal initialization around 0.
        init_weights = torch.from_numpy(np.random.normal(size = (self.n_words,
                                                                 self.word_emb_dim),
                                                         loc = 0.0,
                                                         scale = 0.1
                                                         ).astype(np.float32)
                                        )
        return init_weights

    def _init_normal_par_(self, x, y):
        ### normal initialization around 0.
        init_weights = torch.from_numpy(np.random.normal(size = (x,
                                                                 y),
                                                         loc = 0.0,
                                                         scale = 0.1
                                                         ).astype(np.float32)
                                        )
        return init_weights

    def _init_key_query_normal_(self):
        # normal initialization around 0
        init_weights = torch.from_numpy(np.random.normal(size = (self.n_words,
                                                                 self.key_query_dim),
                                                         loc = 1.0/np.sqrt(self.key_query_dim),
                                                         scale = 0.1
                                                         ).astype(np.float32)
                                        )

        # init_weights = torch.div(torch.ones(self.n_words,self.key_query_dim), np.sqrt(self.key_query_dim))
        return init_weights


class HybridEncoder(nn.Module):
    def __init__(self, cbow_encoder, cmow_encoder):
        super(HybridEncoder, self).__init__()
        self.cbow_encoder = cbow_encoder
        self.cmow_encoder = cmow_encoder

    def forward(self, sent_tuple):
        return torch.cat([self.cbow_encoder(sent_tuple), self.cmow_encoder(sent_tuple)], dim = 1)


def get_cmow_encoder(n_words, padding_idx = 0, word_emb_dim = 784, initialization_strategy = "identity"):
    encoder = Word2MatEncoder(n_words, word_emb_dim = word_emb_dim, 
                              padding_idx = padding_idx, w2m_type = "cmow", 
                              initialization_strategy = initialization_strategy)
    return encoder

def get_cbow_encoder(n_words, padding_idx = 0, word_emb_dim = 784):
    encoder = Word2MatEncoder(n_words, word_emb_dim = word_emb_dim, 
                              padding_idx = padding_idx, w2m_type = "cbow")
    return encoder

def get_cbow_cmow_hybrid_encoder(n_words, padding_idx = 0, word_emb_dim = 400, initialization_strategy = "identity"):
    cbow_encoder = get_cbow_encoder(n_words, padding_idx = padding_idx, word_emb_dim = word_emb_dim)
    cmow_encoder = get_cmow_encoder(n_words, padding_idx = word_emb_dim,
                                   word_emb_dim = word_emb_dim, 
                                   initialization_strategy = initialization_strategy)

    encoder = HybridEncoder(cbow_encoder, cmow_encoder)
    return encoder

def get_acbow_encoder(n_words, padding_idx = 0, word_emb_dim = 784):
    encoder = Word2MatEncoder(n_words, word_emb_dim = word_emb_dim,
                              padding_idx = padding_idx, w2m_type = "acbow")
    return encoder

def get_acbow_encoder_fasttext(n_words, ngram_to_index, wordId2ngramIndices, padding_idx = 0, word_emb_dim = 784):
    encoder = Word2MatEncoder(n_words, word_emb_dim = word_emb_dim,
                              padding_idx = padding_idx, w2m_type = "acbow_fasttext",
                              ngram_to_index=ngram_to_index, wordId2ngramIndices=wordId2ngramIndices)
    return encoder
