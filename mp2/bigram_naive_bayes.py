# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or edoctend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels

def create_vocab_map(train_set, train_label, n_gram):
    pos_word = Counter()
    neg_word = Counter()
    for idx, data_list in enumerate(train_set):
        label = train_label[idx]
        for i in range(len(data_list)-n_gram+1):
            data = "".join(data_list[i:i+n_gram])
            if label == 1:
                pos_word[data] += 1
            else:
                neg_word[data] += 1
    return dict(pos_word), dict(neg_word)

"""
Main function for training and predicting with the bigram midocture model.
    You can modify the default values for the Laplace smoothing parameters, model-midocture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.005, bigram_laplace=0.005, bigram_lambda=0.3, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    pos_map, neg_map = create_vocab_map(train_set, train_labels, 1)
    pos_words = sum(pos_map.values())
    neg_words = sum(neg_map.values())

    pos = pos_words + unigram_laplace * (len(pos_map) + 1)
    neg = neg_words + unigram_laplace * (len(neg_map) + 1)

    pos_bigram_map, neg_bigram_map = create_vocab_map(train_set, train_labels, 2)
    pos_bigram_words = sum(pos_bigram_map.values())
    neg_bigram_words = sum(neg_bigram_map.values())

    pos_bigram = pos_bigram_words + bigram_laplace * (len(pos_bigram_map) + 1)
    neg_bigram = neg_bigram_words + bigram_laplace * (len(neg_bigram_map) + 1)

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        # unigram
        pro_pos = math.log(pos_prior)
        pro_neg = math.log(1 - pos_prior)
        for w in doc:
            if w in pos_map:
                pro_pos += math.log((pos_map[w] + unigram_laplace) / pos)
            else:
                pro_pos += math.log(unigram_laplace / pos)

            if w not in neg_map:
                pro_neg += math.log(unigram_laplace / neg)
            else:
                pro_neg += math.log((neg_map[w] + unigram_laplace) / neg)

        # bigram 
        pro_pos_bigram = math.log(pos_prior)
        neg_prob_bigram = math.log(1 - pos_prior)

        for i in range(len(doc) - 1):
            bigram_word = "".join(doc[i: i + 2])

            if bigram_word not in pos_bigram_map:
                pro_pos_bigram += math.log(bigram_laplace / pos_bigram)
            else:
                pro_pos_bigram += math.log((pos_bigram_map[bigram_word] + bigram_laplace) / pos_bigram)

            if bigram_word in neg_bigram_map:
                neg_prob_bigram += math.log((neg_bigram_map[bigram_word] + bigram_laplace) / neg_bigram)
            else:
                neg_prob_bigram += math.log(bigram_laplace / neg_bigram)

        pro_pos_total = (1 - bigram_lambda) * pro_pos + bigram_lambda * pro_pos_bigram
        neg_prob_total = (1 - bigram_lambda) * pro_neg + bigram_lambda * neg_prob_bigram

        if pro_pos_total > neg_prob_total:
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats




