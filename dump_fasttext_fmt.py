import os
import pickle

import torch
import spacy
from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
import numpy as np

max_ngrams_allowed_per_word = 10
def get_lemma_indices_for_words(word_to_index):
    wordId2lemmaIndices = {}
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    lemmatizer = nlp.vocab.morphology.lemmatizer

    lemma_indices = [0]
    # while len(lemma_indices) < max_ngrams_allowed_per_word:
    #     lemma_indices.append(0)
    wordId2lemmaIndices[0] = lemma_indices

    for word in word_to_index:
        word_index = word_to_index[word]
        lemma_indices = [word_index]

        # noun_x = lemmatizer(word, NOUN)
        adj_x = lemmatizer(word, ADJ)
        verb_x = lemmatizer(word, VERB)
        lemmas_x = [adj_x, verb_x]

        if len(lemmas_x) > 0:
            for elem in lemmas_x:
                for token in elem:
                    if token != word and token in word_to_index and len(lemma_indices) < max_ngrams_allowed_per_word:
                        lemma_indices.append(word_to_index[token])

        # while len(lemma_indices) < max_ngrams_allowed_per_word:
        #     lemma_indices.append(0)

        wordId2lemmaIndices[word_index] = lemma_indices

    return wordId2lemmaIndices


def _save_embeddings_to_word2vec():
    cbow_net = torch.load('/home/user/Code/NLP/awe-project/awe-data/model-data/kq_dim_50_lr_3e4_dim500_acbow_fasttext_v27_model_vecto_500_7L_ep5.cbow_net')
    encoder = cbow_net.module.encoder
    embeddings = encoder.lookup_table
    embeddings = embeddings.weight.data.cpu().numpy()
    word_ngram_idx_table = encoder.word_ngram_idx_table.weight.data.cpu().numpy().astype(int)

    # Load (inverse) vocabulary to match ids to words
    vocabulary = pickle.load(open('/home/user/Code/NLP/awe-project/awe-data/model-data/kq_dim_50_lr_3e4_dim500_acbow_fasttext_v27_model_vecto_500_7L.vocab', 'rb'))[0]
    inverse_vocab = {vocabulary[w]: w for w in vocabulary}

    add_subword = True
    if add_subword:
        # wordId2lemmaIndices = get_lemma_indices_for_words(vocabulary)

        embeddings_subwords = np.zeros((len(inverse_vocab) + 1, embeddings.shape[1]))
        for i in range(1, len(inverse_vocab) + 1):
            # embeddings_subwords[i] = np.mean(np.array([embeddings[idx] for idx in wordId2lemmaIndices[i]]), axis=0)
            embeddings_subwords[i] = np.sum(np.array([embeddings[idx] for idx in word_ngram_idx_table[i]]), axis =0)
            # embeddings_subwords[i] = embeddings_subwords[i] / np.linalg.norm(embeddings_subwords[i])


    # Open file and write values in word2vec format
    output_path = os.path.join('/home/user/Code/NLP/awe-project/awe-data/model-data/kq_dim_50_lr_3e4_dim500_acbow_fasttext_v27_model_vecto_500_7L_ep5_w2v.emb')
    f = open(output_path, 'w')
    print(len(inverse_vocab), embeddings.shape[1], file=f)
    for i in range(1, len(inverse_vocab) + 1):  # skip the padding token
        cur_word = inverse_vocab[i]
        if add_subword:
            f.write(" ".join([cur_word] + [str(embeddings_subwords[i, j]) for j in range(embeddings_subwords.shape[1])]) + "\n")
        else:
            f.write(" ".join([cur_word] + [str(embeddings[i, j]) for j in range(embeddings.shape[1])]) + "\n")

    f.close()

    return output_path


_save_embeddings_to_word2vec()
