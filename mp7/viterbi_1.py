"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = {'START': 1} # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    tag_count = defaultdict(lambda: 0)

    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    for sentence in sentences:
        for i, (word, tag) in enumerate(sentence):
            emit_prob[tag][word] += 1
            tag_count[tag] += 1
            if i == len(sentence) - 1:
                break
            next_tag = sentence[i + 1][1]
            trans_prob[tag][next_tag] += 1

    for i in emit_prob:
        pro = tag_count[i] + emit_epsilon * (len(emit_prob[i]) + 1)
        for j in emit_prob[i]:
            emit_prob[i][j] = (emit_prob[i][j] + emit_epsilon) / pro
        emit_prob[i]['UNSEEN'] = emit_epsilon / pro
   
    for i in trans_prob:
        for j in trans_prob[i]:
            trans_prob[i][j] /= (tag_count[i]-1)

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    tag_set = sorted(list(emit_prob.keys()))
    if i == 0:
        for cur_tag in tag_set:
            if word in emit_prob[cur_tag]:
                log_prob[cur_tag] = log(emit_prob[cur_tag][word]) + prev_prob[cur_tag]
            else:
                log_prob[cur_tag] = log(emit_prob[cur_tag]['UNSEEN']) + prev_prob[cur_tag]
            predict_tag_seq[cur_tag] = [cur_tag]
        return log_prob, predict_tag_seq

    for cur_tag in tag_set:
        op_prevtag = None
        op_log_prob = float('-inf')
        for prev_tag in tag_set:
            prob_trans = log(epsilon_for_pt)
            if cur_tag in trans_prob[prev_tag]:
                prob_trans = log(trans_prob[prev_tag][cur_tag])
            cur_logp = prev_prob[prev_tag] + prob_trans
            if cur_logp > op_log_prob:
                op_prevtag = prev_tag
                op_log_prob = cur_logp
        predict_tag_seq[cur_tag] = list(prev_predict_tag_seq[op_prevtag])
    
        if word in emit_prob[cur_tag]:
            emit_logprob = log(emit_prob[cur_tag][word])
        else:
            emit_logprob = log(emit_prob[cur_tag]['UNSEEN'])
      
        log_prob[cur_tag] = op_log_prob + emit_logprob
        predict_tag_seq[cur_tag] = list(prev_predict_tag_seq[op_prevtag])
        predict_tag_seq[cur_tag].append(cur_tag)

    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        index, max_prob = max(log_prob.items(), key=lambda a: a[1])
        tmp = []
        for j, word in enumerate(sentence):
            tmp.append((word, predict_tag_seq[index][j]))
        predicts.append(tmp)
    return predicts




