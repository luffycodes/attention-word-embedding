import argparse

import torch

from cbow import get_index_batch
from wrap_evaluation import _evaluate_downstream_and_probing_tasks, _get_score_for_name
from torch.autograd import Variable
import pickle

params = {}
params['downstream_eval'] = 'full'


def get_params_parser():
    parser = argparse.ArgumentParser(description='Training a word2mat model.')
    parser.add_argument("--downstream_eval", default="full", type=str, help="Whether to perform 'full'" \
                                                                            "downstream evaluation (slow), 'test' downstream evaluation (fast).",
                        choices=["test", "full"])
    parser.add_argument("--path_to_senteval_dir", type=str,
                        default='/dataroot/attend-word2vec-data/SentEval-master/data', help="path to fastsent data")
    parser.add_argument("--nhid", type=int, default=100,
                        help="Specify the number of hidden units used at test time to train classifiers. If 0 is specified, no hidden layer is employed.")

    parser.add_argument("--downstream_tasks", type=str, nargs="+",
                        default=['OddManOut','CR','MR','SUBJ','MPQA','SST2','SST5','TREC'],
				#'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                                # 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion'],
                        # default=['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                        #     'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                        #     'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                        #     'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
                        #     'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion'],
                        help="Downstream tasks to evaluate on.",
                        choices=['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                                 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                                 'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                                 'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                                 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion'])

    return parser.parse_args()


def prepare(params_senteval, samples):
    params_senteval['vocabulary'] = vocabulary
    params_senteval['inverse_vocab'] = {vocabulary[w]: w for w in vocabulary}


def _batcher_helper(params, batch):
    sent, _ = get_index_batch(batch, vocabulary)
    sent_cuda = Variable(sent.cuda())
    sent_cuda = sent_cuda.t()
    encoder.eval()  # Deactivate drop-out and such
    embeddings = torch.sum(encoder.lookup_table(sent_cuda), dim=1).data.cpu().numpy()
    return embeddings


print("loading vocab")
#vocabulary = pickle.load(open('/dataroot/attend-word2vec-data/tensorboard_vis/acbow_model_784_3.vocab', 'rb'))[0]
vocabulary = pickle.load(open('/dataroot/attend-word2vec-data/saved_full_umbc_models/acbow_model_wiki_400.vocab', 'rb'))[0]
print("loading model")
#cbow_net = torch.load('/dataroot/attend-word2vec-data/tensorboard_vis/acbow_model_784_3.cbow_net')
cbow_net = torch.load('/dataroot/attend-word2vec-data/saved_full_umbc_models/acbow_model_wiki_400.cbow_net')
encoder = cbow_net.module.encoder
print("loaded model")

downstream_scores = _evaluate_downstream_and_probing_tasks(encoder, get_params_parser(), _batcher_helper, prepare)

# from each downstream task, only select scores we care about
to_be_saved_scores = {}
for score_name in downstream_scores:
    to_be_saved_scores[score_name] = _get_score_for_name(downstream_scores, score_name)
print(to_be_saved_scores)
