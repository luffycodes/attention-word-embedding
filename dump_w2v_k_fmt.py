import os
import pickle

import torch


def _save_embeddings_to_word2vec():
    cbow_net = torch.load('/home/user/Code/NLP/awe-project/awe-data/model-data/kq_dim500_mean_root500_std_01_acbow_model_vecto_500_7L_val.cbow_net')
    encoder = cbow_net.module.encoder
    embeddings = encoder.key_table
    embeddings = embeddings.weight.data.cpu().numpy()

    # Load (inverse) vocabulary to match ids to words
    vocabulary = pickle.load(open('/home/user/Code/NLP/awe-project/awe-data/model-data/kq_dim500_mean_root500_std_01_acbow_model_vecto_500_7L.vocab', 'rb'))[0]
    inverse_vocab = {vocabulary[w]: w for w in vocabulary}

    # Open file and write values in word2vec format
    output_path = os.path.join('/home/user/Code/NLP/awe-project/awe-data/kq_dim500_mean_root500_std_01_acbow_model_vecto_500_7L_val_ep_pt4_key.emb')
    f = open(output_path, 'w')
    print(embeddings.shape[0] - 1, embeddings.shape[1], file=f)
    for i in range(1, embeddings.shape[0]):  # skip the padding token
        cur_word = inverse_vocab[i]
        f.write(" ".join([cur_word] + [str(embeddings[i, j]) for j in range(embeddings.shape[1])]) + "\n")

    f.close()

    return output_path


_save_embeddings_to_word2vec()
