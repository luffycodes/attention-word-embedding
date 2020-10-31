import pickle
import torch
import numpy as np
import spacy
from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
import torch.nn as nn


def _save_embeddings_to_word2vec():
    cbow_net = torch.load('/home/user/Code/NLP/awe-project/awe-data/model-data/kq_dim_50_lr_3e4_dim500_acbow_fasttext_v24_model_vecto_500_7L_ep2.cbow_net')
    encoder = cbow_net.module.encoder
    key_embeddings = encoder.key_table
    # key_embeddings.weight.div_(torch.norm(key_embeddings.weight, p=2, dim=1).unsqueeze_(-1))
    query_embeddings = encoder.query_table
    value_embeddings = encoder.lookup_table
    # key_embeddings = key_embeddings.weight.data.cpu().numpy()

    # Load (inverse) vocabulary to match ids to words
    vocabulary = pickle.load(open('/home/user/Code/NLP/awe-project/awe-data/model-data/kq_dim_50_lr_3e4_dim500_mean_root500_std_dot1_acbow_model_vecto_500_7L.vocab', 'rb'))[0]
    inverse_vocab = {vocabulary[w]: w for w in vocabulary}

    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    lemmatizer = nlp.vocab.morphology.lemmatizer

    count = 0
    expand_vocab = 0
    lemmas_ = set()
    for x in vocabulary.keys():
        noun_x = lemmatizer(x, NOUN)
        adj_x = lemmatizer(x, ADJ)
        verb_x = lemmatizer(x, VERB)
        lemmas_x = {noun_x[0], adj_x[0], verb_x[0]}
        lemmas_x.remove(x)

        if len(lemmas_x) > 0:
            # print(x, lemmas_x)
            for elem in lemmas_x:
                lemmas_.add(elem)
            expand_vocab += len(lemmas_x)
        count += 1
        if count % 100000 == 0:
            print("------------------------------", count, expand_vocab)


    words = ['tea', 'book', 'monitor', 'anxious', 'phone', 'fish', 'gambling', 'professor']
    for word in words:
        word_ = key_embeddings.weight[vocabulary[word]]
        pred_vect = torch.matmul(query_embeddings.weight, word_)
        # top_pred_word_ind = np.argpartition(pred_vect.data.cpu().numpy(), -20)[-20:]
        top_pred_word_ind = np.argsort(pred_vect.data.cpu().numpy())[-10:]
        top_pred_word = [inverse_vocab[x] for x in top_pred_word_ind]
        print(word, top_pred_word)

    while True:
        word = input("enter word\n")
        sent = input("enter sent\n")

        key_word_emb = key_embeddings.weight[vocabulary[word]]
        # key_word_emb = torch.div(key_word_emb, torch.norm(key_word_emb))
        value_word_emb = value_embeddings.weight[vocabulary[word]]
        # value_word_emb = torch.div(value_word_emb, torch.norm(value_word_emb))

        for query_word in sent.split():
            query_word_emb = query_embeddings.weight[vocabulary[query_word]]
            # query_word_emb = torch.div(query_word_emb, torch.norm(query_word_emb))

            value_query_word_emb = value_embeddings.weight[vocabulary[query_word]]
            # value_query_word_emb = torch.div(value_query_word_emb, torch.norm(value_query_word_emb))

            # print(query_word, round(torch.exp(torch.dot(key_word_emb, query_word_emb)).item(), 3), round(torch.norm(query_word_emb, p=2).item(), 2))
            print(query_word, round(torch.exp(torch.dot(key_word_emb, query_word_emb)).item(), 3), round(torch.dot(value_word_emb, value_query_word_emb).item(), 3))


_save_embeddings_to_word2vec()
