import pickle
import torch
import numpy as np


def _save_embeddings_to_word2vec():
    cbow_net = torch.load('/home/user/Code/NLP/awe-project/awe-data/model-data/kqv_lr_5e4_decay_y_09_dim500_mean_root500_std_01_acbow_vecto_500_7L_ep2.cbow_net')
    encoder = cbow_net.module.encoder
    key_embeddings = encoder.key_table
    query_embeddings = encoder.query_table
    # key_embeddings = key_embeddings.weight.data.cpu().numpy()

    # Load (inverse) vocabulary to match ids to words
    vocabulary = pickle.load(open('/home/user/Code/NLP/awe-project/awe-data/model-data/kqv_lr_5e4_decay_y_09_dim500_mean_root500_std_01_acbow_vecto_500_7L.vocab', 'rb'))[0]
    inverse_vocab = {vocabulary[w]: w for w in vocabulary}

    words = ['tea', 'book', 'monitor', 'anxious', 'phone', 'fish', 'gambling', 'professor']
    for word in words:
        word_ = key_embeddings.weight[vocabulary[word]]
        pred_vect = torch.matmul(query_embeddings.weight, word_)
        # top_pred_word_ind = np.argpartition(pred_vect.data.cpu().numpy(), -20)[-20:]
        top_pred_word_ind = np.argsort(pred_vect.data.cpu().numpy())[-20:]
        top_pred_word = [inverse_vocab[x] for x in top_pred_word_ind]
        print(word, top_pred_word)

_save_embeddings_to_word2vec()
