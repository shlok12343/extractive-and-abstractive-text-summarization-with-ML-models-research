# !pip install rouge_score

# imports
import pandas as pd
import numpy as np
import torch
from rouge_score import rouge_scorer

from clean_preprocess import (
    clean_transcript_lang,
    merge_short_sentences,
    split_sentences
)

from ffn_summ import *

# retrains models - see line 22
# from log_reg_and_ffn import SUMMARY_MAX_LEN, device

import pickle

# use these instead
SUMMARY_MAX_LEN = 10  # same as in your training script
device = 'cuda' if torch.cuda.is_available() else 'cpu'

df_transcript = pd.read_pickle('src/pkl_files/df_transcript.pkl')
df_summ = pd.read_pickle('src/pkl_files/df_summ.pkl')

with open('src/pkl_files/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('src/pkl_files/log_reg_model.pkl', 'rb') as f:
    log_reg_model = pickle.load(f)

ids_test = np.load('src/npy_files/ids_test.npy')
input_dim = int(np.load('src/npy_files/input_dim.npy')[0])

# rebuild FFN, load weights from .pth
feedforward_net_summary = FF_Net_Summary(input_dim).to(device)
state_dict = torch.load('src/ffn_pth/ffn_summary.pth', map_location=device)
feedforward_net_summary.load_state_dict(state_dict)
feedforward_net_summary.eval()

# see summarization in action
def summarize_lecture(transcript, tfidf, log_reg_model, nn_model,
                      max_sentences=SUMMARY_MAX_LEN):
    # clean and split transcript
    transcript_clean = clean_transcript_lang(transcript.lower())
    sentences = merge_short_sentences(split_sentences(transcript_clean))

    # vectorize
    X = tfidf.transform(sentences)

    # logistic regression
    probs_log_reg = log_reg_model.predict_proba(X)[:, 1]
    # choose top-K sentences
    k = min(max_sentences, len(sentences))
    idxs_log_reg = np.argsort(probs_log_reg)[-k:]
    idxs_log_reg = np.sort(idxs_log_reg)  # keep original order
    summary_log_reg = [sentences[i] for i in idxs_log_reg]

    # NN
    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32).to(device)
    nn_model.eval()
    with torch.no_grad():
        outputs = nn_model(X_tensor).squeeze(1)
        probs_nn = torch.sigmoid(outputs).cpu().numpy()
    idxs_nn = np.argsort(probs_nn)[-k:]
    idxs_nn = np.sort(idxs_nn)
    summary_nn = [sentences[i] for i in idxs_nn]

    return sentences, summary_log_reg, summary_nn

# output ROUGE scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge_log_reg = []
rouge_nn = []

for id_script in ids_test:
    # get transcript and hand summary
    transcript = df_transcript.loc[df_transcript['transcript_id']==id_script, 'transcript_clean'].iloc[0]
    hand_summ = df_summ.loc[df_summ['transcript_id']==id_script, 'paraphrased_summary'].iloc[0]

    # generate summaries
    sentences, summary_log_reg, summary_nn = summarize_lecture(transcript, tfidf, log_reg_model, feedforward_net_summary)

    log_s_text = " ".join(summary_log_reg)
    nn_s_text  = " ".join(summary_nn)

    # compute scores
    rouge_log_reg.append(scorer.score(hand_summ, log_s_text))
    rouge_nn.append(scorer.score(hand_summ, nn_s_text))

# output ROUGE scores
print (rouge_log_reg)

print (rouge_nn)

# genetic algorithms lecture text
ga_text = """
Genetic Algorithms

Genetic Algorithms are a class of search algorithms inspired by the process of natural selection, and can be viewed as an extension/combination of some of the local search techniques we have discussed so far. The algorithm, much like local beam search, maintains a population of candidate solutions, and iteratively applies genetic operators to the population to generate new candidate solutions. These genetic operators broadly consist of mutations and crossovers, discussed below. The algorithm proceeds by iteratively generating new candidates using these operators, and then using a fitness function to retain some subset of the best-performing solutions, in a workflow similar to local beam search

Mutations: These are random changes to a candidate solution, just like in regular hill climbing. For example, in the case of a binary string, a mutation could involve flipping a single bit from 0 to 1 or vice-versa. In the case of a real-valued vector, a mutation could involve adding a small random number to one of the vector's elements. In the case of the traveling salesman problem, a 2-opt swap could be a mutation. Mutations are used to introduce new information into the population, and to prevent the algorithm from getting stuck in local optima. The rate at which these random changes are applied to each candidate solution is controlled by a parameter called the mutation rate, and may vary throughout the execution of the program (similar to how we decrease probability of large changes over time in Simulated Annealing). Crossovers: These are operations that combine two candidate solutions to produce one or more new candidate solutions. For example, in the case of a binary string or a real-valued 1-dimensional vector, one possible crossover could involve taking the first half of one parent and the second half of another parent to produce a new child string/vector. In the case of the traveling salesman problem, crossovers might involve more steps and some logical parsing of the two parent solutions (see below). Crossovers are used to combine information from two candidate solutions that are already good, in the hopes of producing an even better solution. Complex crossovers may often be necessary, especially in cases where the validity of a solution depends on the structure of its representation. For example, consider the traveling salesman problem with 5 cities, named A, B, C, D and E. Assuming all cities are connected, one possible solution could be the string "ABCDEA", which represents the order in which the cities are visited. A second candidate solution could be AECDBA. Now, let's try and combine these solutions using a crossover operation. A simple crossover between the two parents "ABCDEA" and "AECDBA" where we combine the first half of the first parent with the second half of the second parent would yield the string "ABCDBA", which is not a valid solution to the TSP. This is because the child visits city B twice, and does not visit city E at all. A more complex crossover could involve parsing the two parents and combining them in a way that ensures that the child visits each city exactly once. This is known as the Order Crossover (OX) operator.
"""

sentences_ga, ga_log_reg, ga_nn = summarize_lecture(
    ga_text,
    tfidf,
    log_reg_model,
    feedforward_net_summary,
    max_sentences=SUMMARY_MAX_LEN
)

print('Original Excerpt:\n')
for s in sentences_ga:
    print(" -", s)

print('\nLogistic Regression Summary:\n')
for s in ga_log_reg:
    print(s)

print('\nNeural Net Summary:\n')
for s in ga_nn:
    print(s)