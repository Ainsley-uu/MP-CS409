# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
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
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_vocab_map(train_set, train_label):
    pos_word = Counter()
    neg_word = Counter()
    for idx, data_list in enumerate(train_set):
        label = train_label[idx]
        for data in data_list:
            if label == 1:
                pos_word[data] += 1
            else:
                neg_word[data] += 1
    return dict(pos_word), dict(neg_word)


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=0.001, pos_prior=0.5, silently=False):
    pos_map, neg_map = create_vocab_map(train_set, train_labels)
    pos_words = sum(pos_map.values())
    neg_words = sum(neg_map.values())

    pos = pos_words + laplace * (len(pos_map) + 1)
    neg = neg_words + laplace * (len(neg_map) + 1)

    yhats = []
    for data in tqdm(dev_set):
        pos_pro = math.log(pos_prior)
        neg_pro = math.log(1 - pos_prior)
        for d in data:
            if d not in pos_map:
                pos_pro += math.log(laplace / pos)
            else: 
                pos_pro += math.log((pos_map[d] + laplace) / pos) 

            if d not in neg_map:
                neg_pro += math.log(laplace / neg) 
            else:
                neg_pro += math.log((neg_map[d] + laplace) / neg)

        if pos_pro < neg_pro:
            yhats.append(0)
        else:
            yhats.append(1)
    return yhats

# DECLARE ScaledSPI REAL;
# SPIValues AS (
#     SELECT
#         Department,
#         NetId,
#         (SUM(Score * Credits) OVER (PARTITION BY Department) / SUM(Credits) OVER (PARTITION BY Department)) AS WeightedAvg,
#         MIN(NumCourses) OVER () AS MinCourses,
#         COUNT(DISTINCT NetId) OVER (PARTITION BY Department) AS DistinctStudents
#     FROM FilteredScores
# )
select c.Department, count(s.NetId) as NumStudents, 
sum(e.Score * e.Credits)/sum(e.credits) * (NumStudents/min(e.Credits)) as ScaledSPI
from Courses c 
    left join Enrollments e on c.CRN = e.CRN
    left join Students s on s.NetId = e.NetId
order by NumStudents, ScaledSPI decs