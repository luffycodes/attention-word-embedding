"""
This file handles everything related to the CBOW task, i.e., it creates the training examples (CBOWDataset), and provides the neural architecture (except encoder) and loss computation (see CBOWNet).
"""

import os, pickle, math
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from collections import Counter
import nltk.data
from nltk.tokenize import word_tokenize
from random import shuffle
import random
import json

import torch.nn as nn
from torch import FloatTensor as FT
from torch import ByteTensor as BT
from torch.autograd import Variable
import spacy
from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
max_ngrams_allowed_per_word = 10

def recursive_file_list(path):
    """
    Recursively aggregates all files at the given path, i.e., files in subfolders are also
    included.
    """
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn if ".txt" in f]

def tokenize(sent, sophisticated = False):
    """
    Tokenizes the sentence. If 'sophisticated' is set to False, the
    tokenization is a simple split by the blank character. Otherwise the
    TreebankWordTokenizer provided by NLTK.
    """
    return sent.split() if not sophisticated else word_tokenize(sent)

def sentenize(text):
    return sent_tokenizer.tokenize(text)

def get_wordvec_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths

def get_index_batch(batch, word_vec):

    # remove all words that are out of vocabulary
    clean_batch = []
    for sen in batch:
        clean_batch.append([w for w in sen if w in word_vec])
    batch = clean_batch
    
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch)))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).long(), lengths


def get_word_ngrams(word, minn=3, maxn=4):
    word_ngrams_set = set()
    word_len = len(word)
    for n in range(minn, maxn + 1):
        for i in range(word_len - n + 1):
            ngram = word[i:i + n]
            if len(ngram) == word_len:
                continue
            elif len(ngram) > 1 and i == 0:
                word_ngrams_set.add('<'+ngram)
            elif len(ngram) > 1 and i == word_len -n:
                word_ngrams_set.add(ngram + '>')
            else:
                word_ngrams_set.add(ngram)
    return word_ngrams_set


def get_ngram_dict_for_words(words, minn=3, maxn=4):
    ngram_dict = {}
    word2ngram_dict = {}
    words_processed = 0
    for word in words:
        word2ngram_dict[word] = get_word_ngrams(word, minn, maxn)
        for ngram in word2ngram_dict[word]:
            if ngram not in ngram_dict:
                ngram_dict[ngram] = 1
            else:
                ngram_dict[ngram] += 1

        words_processed += 1
        if words_processed % 100000 == 0:
            print('size_ngram_dict:', words_processed, len(ngram_dict))

    return ngram_dict, word2ngram_dict


def get_lemma_indices_for_words(word_to_index):
    wordId2lemmaIndices = {}
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    lemmatizer = nlp.vocab.morphology.lemmatizer

    lemma_indices = [0]
    while len(lemma_indices) < max_ngrams_allowed_per_word:
        lemma_indices.append(0)
    wordId2lemmaIndices[0] = lemma_indices

    for word in word_to_index:
        word_index = word_to_index[word]
        lemma_indices = [word_index]

        noun_x = lemmatizer(word, NOUN)
        adj_x = lemmatizer(word, ADJ)
        verb_x = lemmatizer(word, VERB)
        lemmas_x = [noun_x, adj_x, verb_x]

        if len(lemmas_x) > 0:
            for elem in lemmas_x:
                for token in elem:
                    if token != word and token in word_to_index and len(lemma_indices) < max_ngrams_allowed_per_word:
                        lemma_indices.append(word_to_index[token])

        while len(lemma_indices) < max_ngrams_allowed_per_word:
            lemma_indices.append(0)

        wordId2lemmaIndices[word_index] = lemma_indices

    return wordId2lemmaIndices


def get_ngram_indices_for_words(word2ngram_dict, word_to_index, ngram_to_index):
    wordId2ngramIndices = {}
    ngram_indices = [0]
    while len(ngram_indices) < max_ngrams_allowed_per_word:
        ngram_indices.append(0)
    wordId2ngramIndices[0] = ngram_indices

    for word in word_to_index:
        word_index = word_to_index[word]
        ngram_indices = [word_index]
        for ngram in word2ngram_dict[word]:
            if ngram in ngram_to_index:
                if len(ngram_indices) < max_ngrams_allowed_per_word:
                    ngram_indices.append(ngram_to_index[ngram])

        while len(ngram_indices) < max_ngrams_allowed_per_word:
            ngram_indices.append(0)

        wordId2ngramIndices[word_index] = ngram_indices
    return wordId2ngramIndices


def get_word_dict(sentences):
    # create vocab of words and also count occurences
    word_dict = {}
    for sent in sentences:
        for word in tokenize(sent):
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1

    return word_dict


def get_wordembedding(word_dict, we_path):
    # create word_vec with glove vectors
    word_voc = {}
    with open(we_path) as f:

        # discard the information in first row
        _, emb_size = f.readline().split()

        i = 1
        word_embs = []
        for line in f:
            line = line.strip('\n').split()
            word_end = len(line) - int(emb_size)
            word = " ".join(line[:word_end])
            if word in word_dict:
                word_voc[word] = i
                word_embs.append(np.asarray(list(map(float, line[word_end:]))))
                i += 1
    print('Found {0}(/{1}) words with glove vectors'.format(
                len(word_voc), len(word_dict)))

    word_embs = np.vstack(word_embs)
    word_count = {w : word_dict[w] for w in word_voc}
    return word_voc, word_count, word_embs

def get_index_vocab(word_dict, max_words):
    if max_words is not None:
        counter = Counter(word_dict)
        most_common_words = counter.most_common(max_words)
        reduced_word_dict = {}
        for w, cnt in most_common_words:
            reduced_word_dict[w] = cnt
        word_dict = reduced_word_dict
    print('Num words in corpus : {:,}'.format(np.sum([word_dict[w] for w in word_dict])))

    # create word_vec with glove vectors
    word_vec = {}
    idx = 1 # reserve 0 for padding_idx
    for word in word_dict:
        word_vec[word] = idx
        idx += 1
    return word_vec, word_dict


def build_vocab(sentences, minn, maxn, max_ngrams, append_ngrams, pretrained_embeddings = None, max_words = None):
    word_dict = get_word_dict(sentences)
    if pretrained_embeddings:
        word_to_index, word_to_count, word_embeddings = get_wordembedding(word_dict, pretrained_embeddings)
    else:
        word_to_index, word_to_count = get_index_vocab(word_dict, max_words) # padding_idx = 0
    print('Vocab size : {0}'.format(len(word_to_index)))

    ngram_to_index ={}
    wordId2ngramIndices = {}
    if append_ngrams:
        ngram_dict, word2ngram_dict = get_ngram_dict_for_words(word_to_index, minn, maxn)
        ngram_to_index, ngram_to_count = get_index_vocab(ngram_dict, max_ngrams)
        for key in ngram_to_index:
            ngram_to_index[key] += max_words + 1
        with open('ngram_to_count.json', 'w') as fp:
            json.dump(ngram_to_count, fp)
            print("ngram to count dumped")
        with open('ngram_to_index.json', 'w') as fp:
            json.dump(ngram_to_index, fp)
            print("ngram to index dumped")
        # debug - list(word_to_index.keys())[list(word_to_index.values()).index(16)]
        wordId2ngramIndices = get_ngram_indices_for_words(word2ngram_dict, word_to_index, ngram_to_index)
        wordId2lemmaIndices = get_lemma_indices_for_words(word_to_index)


    if pretrained_embeddings:
        return word_to_index, word_to_count, word_embeddings
    else:
        return word_to_index, word_to_count, ngram_to_index, wordId2lemmaIndices

class CBOWDataset(Dataset):
    """
    Considers each line of a file to be a text.
    Reads all files found at directory 'path' and corresponding subdirectories.
    """
    def __init__(self, path, num_texts, context_size, num_samples_per_item, mode, precomputed_word_vocab, max_words, pretrained_embeddings, num_texts_per_chunk, precomputed_chunks_dir, temp_path,
                 minn=3, maxn=4, max_ngrams=10000, append_ngrams=True):

        self.context_size = context_size
        self.num_samples_per_item = num_samples_per_item
        self.mode = mode
        self.num_texts_per_chunk = num_texts_per_chunk

        texts_generator = _generate_texts(path, num_texts)

        # load precomputed word vocabulary and counts
        if precomputed_word_vocab:
            word_vec = pickle.load(open(os.path.join(precomputed_word_vocab), "rb" ))
            self._word_vec_count_tuple = word_vec
            self.word_vec, self.word_count = word_vec
        else:
            self.word_vec, self.word_count, self.ngram_to_index, self.wordId2ngramIndices = build_vocab(texts_generator,
                                   minn, maxn, max_ngrams, append_ngrams,
                                   pretrained_embeddings = pretrained_embeddings,
                                   max_words = max_words)
            self._word_vec_count_tuple = self.word_vec, self.word_count

        # create chunks
        self.num_texts = num_texts
        self.num_chunks = math.ceil(num_texts / (1.0*self.num_texts_per_chunk))
        self._temp_path = temp_path
        if not os.path.exists(self._temp_path):
            os.makedirs(self._temp_path)

        if precomputed_chunks_dir is None:
            self._create_chunk_files(_generate_texts(path, num_texts))
            self._check_chunk_files()
        else:
            self._temp_path = precomputed_chunks_dir
            print("use precomputed chunk files.")

        # self._word_vec_count_tuple = word_vec
        # self.word_vec, self.word_count = word_vec
        self.num_training_samples = self.num_texts

        # compute unigram distribution
        ## set frequency of padding token to 0 implicitly
        unigram_dist = np.zeros((len(self.word_vec) + 1))
        for w in self.word_vec:
            unigram_dist[self.word_vec[w]] = self.word_count[w]

        self.unigram_dist = unigram_dist
        # sum_unigram_dist = np.sum(self.unigram_dist)
        # self.unigram_dist_prob = [1 - np.sqrt(0.00001 * sum_unigram_dist / x) for x in self.unigram_dist]


    def _count_words_per_text(self):
        text_lengths = [0] * len(self.texts)

        for i, text in enumerate(self.texts):

            words = tokenize(text)
            words = [self.word_vec[w] for w in words if w in self.word_vec]
            text_lengths[i] = len(words)

        return text_lengths

    def _check_chunk_files(self):
        """Raises an exception if any of the chunks generated 
        is empty.
        """
        for i in range(self.num_chunks):
            with open(self._get_chunk_file_name(i), "r") as f:
                lines = f.readlines()
                if(len(lines) == 0):
                    raise Exception("Chunk ", i, " is empty\n")

    def _create_chunk_files(self, texts_generator):
        cur_chunk_number = 0
        cur_chunk_file = open(self._get_chunk_file_name(cur_chunk_number), "w")
        cur_idx = 0
        last_chunk_size = self.num_texts - (self.num_texts_per_chunk*(self.num_chunks-1))
        for text in texts_generator:
            print(text, file=cur_chunk_file)
            if cur_idx == self.num_texts_per_chunk - 1 or (cur_idx == last_chunk_size-1 and 
                    cur_chunk_number == self.num_chunks-1):
                # start next chunk
                cur_chunk_file.close()
                cur_idx = 0  # index within the chunk
                cur_chunk_number += 1
                cur_chunk_file = open(self._get_chunk_file_name(cur_chunk_number), "w")
            else:
                cur_idx += 1
        cur_chunk_file.close()

    def _get_chunk_file_name(self, chunk_number):
        return os.path.join(self._temp_path, "chunk" + str(chunk_number))

    def __len__(self):
        return self.num_texts

    def _load_text(self, idx):
        chunk_number = math.floor(idx / (1.0*self.num_texts_per_chunk))
        idx_in_chunk = idx % self.num_texts_per_chunk
        with open(self._get_chunk_file_name(chunk_number), "r") as f:
            for i, line in enumerate(f):
                if i == idx_in_chunk:
                    return line.strip()
        raise Exception("Text with idx: ", idx, " in chunk: ", chunk_number,\
                " and idx_in_chunk: ", idx_in_chunk, " not found.")

    def _compute_idx_to_text_word_dict(self):
        idx_to_text_word_tuple = {}
        idx = 0
        for i, text in enumerate(self.texts):
            
            for j in range(self.text_lengths[i]):
                idx_to_text_word_tuple.update({idx : (i, j)})
                idx += 1

        self.idx_to_text_word_tuple = idx_to_text_word_tuple

    def _create_window_samples(self, words):
        # words = [x for x in words if np.random.random_sample() > self.unigram_dist_prob[x] and x != 0]
        words = [x for x in words if x != 0]

        if len(words) == 0:
            return None, None

        text_len = len(words)
        num_samples = min(text_len, self.num_samples_per_item)

        words = [0] * self.context_size + words + [0] * self.context_size

        training_sequences = np.zeros((num_samples, 2 * self.context_size, 1))
        missing_words = np.zeros((num_samples))

        training_sequences_sg = []
        missing_words_sg = []

        # randomly select mid_words to use
        count = 0
        mid_words = random.sample(range(text_len), num_samples)
        for i, j in enumerate(mid_words):

            middle_word = self.context_size + j

            # choose a word that is removed from the window
            if self.mode == 'random':
                rand_offset = random.randint(-self.context_size, self.context_size)
                missing_word = middle_word + rand_offset
            elif self.mode == 'cbow':
                missing_word = middle_word
            else:
                raise NotImplementedError("Unknown training mode " + self.mode)

            # zero is the padding word
            training_sequence = [middle_word + context_word for context_word in range(-self.context_size, self.context_size + 1) if middle_word + context_word != missing_word]
            training_sequence = [words[w] for w in training_sequence]
            training_sequences_ngram = [[self.wordId2ngramIndices[word][0]] for word in training_sequence]
            training_sequences[i, :] = np.array(training_sequences_ngram)
            missing_word = words[missing_word]
            missing_words[i] = np.array(missing_word)

        #     for word in training_sequence:
        #         if word != 0 and missing_word != 0:
        #             training_sequences_sg.append(np.array([self.wordId2ngramIndices[word]]))
        #             missing_words_sg.append([missing_word])
        #             count = 1
        #
        # if count == 0:
        #     return None, None
        #
        # return np.array(training_sequences_sg), np.array(missing_words_sg)
        return training_sequences, missing_words

    def __getitem__(self, idx):

        text = self._load_text(idx)
        words = tokenize(text)
        words = [self.word_vec[w] for w in words if w in self.word_vec]
        text_len = len(words)

        # TODO: is there a better way to handle empty texts?
        if text_len == 0:
            return None, None

        if self.mode in ['random', 'cbow']:
            return self._create_window_samples(words)
        else:
            raise NotImplementedError("Unknown mode " + str(self.mode))

    ## collate function for cbow
    def collate_fn(self, l):
        l1, l2 = zip(*l)
        l1 = [x for x in l1 if x is not None]
        l2 = [x for x in l2 if x is not None]
        l1 = np.vstack(l1)
        l2 = np.concatenate(l2)
        return torch.from_numpy(l1).long(), torch.from_numpy(l2).long()

def _load_texts(path, num_docs):
    texts = []
    filename_list = recursive_file_list(path)
    
    for filename in filename_list:
        if filename.endswith(".txt"):
            with open(os.path.realpath(filename), 'r') as f:

                # change encoding to utf8 to be consistent with other datasets
                #cur_text.decode("ISO-8859-1").encode("utf-8")
                for line in f:
                    line = line.strip()
                    texts.append(line)

                    if num_docs is not None and len(texts) > num_docs:
                        break

    return texts

def _generate_texts(path, num_docs):
    filename_list = recursive_file_list(path)

    for filename in filename_list:
        with open(os.path.realpath(filename), "r") as f:

            # change encoding to utf8 to be consistent with other datasets
            # cur_text.decode("ISO-8859-1").encode("utf-8")
            for i, line in enumerate(f):
                line = line.strip()
                if num_docs is not None and i > num_docs - 1:
                    break
                yield line

class CBOWNet(nn.Module):
    def __init__(self, encoder, output_embedding_size, output_vocab_size, weights = None, n_negs = 20, padding_idx = 0):
        super(CBOWNet, self).__init__()

        self.encoder = encoder
        self.n_negs = n_negs
        self.weights = weights
        self.output_vocab_size = output_vocab_size
        self.output_embedding_size = output_embedding_size

        # self.outputembeddings = nn.Embedding(output_vocab_size + 1, output_embedding_size, padding_idx=0)
        self.outputembeddings = encoder.lookup_table

        self.w2m_type = encoder.w2m_type
        if self.w2m_type == "acbow_fasttext":
            self.word_ngram_idx_table = encoder.word_ngram_idx_table

        # self.V = nn.Embedding(output_vocab_size + 1, output_embedding_size, padding_idx=0)

        if self.weights is not None:
            wf = np.power(self.weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, input_s, missing_word):

        embedding = self.encoder(input_s, missing_word)
        batch_size = embedding.size()[0]
        emb_size = embedding.size()[1]
        
        # draw negative samples
        if self.weights is not None:
            nwords = torch.multinomial(self.weights, batch_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, self.n_negs).uniform_(0, self.vocab_size).long()
        nwords = Variable(torch.LongTensor(nwords), requires_grad=False).cuda()       

        # lookup the embeddings of output words
        if self.w2m_type != "acbow_fasttext":
            missing_word_vector = self.outputembeddings(missing_word)
        else:
            missing_word_ngram_indices = self.word_ngram_idx_table(missing_word).long()
            missing_word_vector = torch.sum(self.outputembeddings(missing_word_ngram_indices), dim=1)

            # missing_mask_sum = (missing_word_ngram_indices != 0).sum(dim=1).clamp(min=1)
            # missing_mask_sum = missing_mask_sum.unsqueeze_(-1)
            #
            # missing_word_vector.div_(missing_mask_sum)

        if self.w2m_type != "acbow_fasttext":
            nvectors = self.outputembeddings(nwords).neg()
        else:
            nwords_ngram_indices = self.word_ngram_idx_table(nwords).long()
            nvectors = torch.sum(self.outputembeddings(nwords_ngram_indices), dim=2).neg()

            # nwords_mask_sum = (nwords_ngram_indices != 0).sum(dim=2).clamp(min=1)
            # nwords_mask_sum = nwords_mask_sum.unsqueeze_(-1)
            #
            # nvectors.div_(nwords_mask_sum)
            # nvectors.neg_()

        # compute loss for correct word
        oloss = torch.bmm(missing_word_vector.view(batch_size, 1, emb_size), embedding.view(batch_size, emb_size, 1))
        oloss = oloss.squeeze().sigmoid()

        ## add epsilon to prediction to avoid numerical instabilities
        oloss = self._add_epsilon(oloss)
        oloss = oloss.log()
        
        # compute loss for negative samples
        nloss = torch.bmm(nvectors, embedding.view(batch_size, -1, 1)).squeeze().sigmoid()

        ## add epsilon to prediction to avoid numerical instabilities
        nloss = self._add_epsilon(nloss)
        nloss = nloss.log()
        nloss = nloss.mean(1)

        # combine losses
        return -(oloss + nloss)

    def _add_epsilon(self, pred):
        return pred + 0.00001

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb

