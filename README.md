# Attention Word Embeddings

The code is inspired from the following [github repository](https://github.com/florianmai/word2mat). 

*AWE* is designed to learn rich word vector representations. It fuses the attention mechanism with the CBOW model of word2vec to address the limitations of the CBOW model. CBOW equally weights the context words when making the masked word prediction, which is inefficient, since
some words have higher predictive value than others. We tackle this inefficiency by introducing
our Attention Word Embedding (*AWE*) model. We also propose AWE-S, which incorporates subword information (code for which is in the fastText branch).

Details of this method and results can be found in our [COLING PAPER](https://arxiv.org/pdf/2006.00988.pdf).
