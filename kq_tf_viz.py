# Tutorial link:  https://github.com/sudharsan13296/Hands-On-Deep-Learning-Algorithms-with-Python/blob/master/07.%20Learning%20Text%20Representations/7.08%20Visualizing%20Word%20Embeddings%20in%20TensorBoard.ipynb

import warnings

warnings.filterwarnings(action='ignore')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorboard.plugins import projector

import os
import torch
import pickle
import numpy as np

from sklearn.decomposition import PCA


def vis_w2v():
    cbow_net = torch.load(
        '/home/user/Code/NLP/awe-project/awe-data/model-data/kq_dim_50_lr_3e4_dim500_mean_root500_std_01_acbow_model_vecto_500_7L_ep5.cbow_net')
    encoder = cbow_net.module.encoder
    key_embeddings = encoder.key_table
    query_embeddings = encoder.query_table
    n_words = 100000
    kq_embeddings = torch.matmul(key_embeddings.weight[0:n_words], query_embeddings.weight[0:n_words].T)
    kq_embeddings = kq_embeddings.detach()
    torch.exp(kq_embeddings, out=kq_embeddings)
    kq_embeddings = kq_embeddings.data.cpu().numpy()

    # Load (inverse) vocabulary to match ids to words
    vocabulary = pickle.load(open(
        '/home/user/Code/NLP/awe-project/awe-data/model-data/kq_dim_50_lr_3e4_dim500_mean_root500_std_01_acbow_model_vecto_500_7L.vocab',
        'rb'))[0]
    inverse_vocab = {vocabulary[w]: w for w in vocabulary}

    if not os.path.exists('projections'):
        os.makedirs('projections')

    max_size = kq_embeddings.shape[0]
    max_words = min(kq_embeddings.shape[0], 100000) - 1
    w2v = np.zeros((max_words, kq_embeddings.shape[1]))
    with open("projections/metadata.tsv", 'w+') as file_metadata:
        for i in range(0, max_words):
            file_metadata.write(inverse_vocab[i+1] + '\n')
            w2v[i] = kq_embeddings[i+1]

    pca = PCA(n_components=50)
    pca.fit(w2v)
    w2v = pca.transform(w2v)

    sess = tf.compat.v1.InteractiveSession()
    with tf.device("/cpu:0"):
        embedding = tf.Variable(w2v, trainable=False, name='embedding')
    tf.compat.v1.global_variables_initializer().run()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('projections', sess.graph)
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'embedding'
    embed.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(writer, config)

    saver.save(sess, 'projections/model.ckpt', global_step=max_size)


vis_w2v()
