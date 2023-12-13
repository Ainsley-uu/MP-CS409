"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""
import numpy as np
from collections import defaultdict
import math

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    result = []
    arg1, word_bag, arg2, arg3, arg4 = training(train)
    init_prob, trans_prob, emi_prob, emi_hapax_prob = smoothing(arg1, word_bag, arg2, arg3, arg4)

    tag_list = list(init_prob.keys())
    count = 0
    tag_num = len(init_prob)

    X_ING = "X_ING"
    X_ED = "X_ED"
    X_LY = "X_LY"
    UN_X = "UN_X"
    IN_X = "IN_X"

    special_fix_list = [X_ING,X_ED,X_LY,UN_X,IN_X]
    special_fix_tag_count = {} # special_fix_tag_count[tag][fix] = occ 
    special_fix_tag_prob = {} # special_fix_tag_prob[tag][fix] = probability
    for tag in tag_list:
        special_fix_tag_count[tag] = {X_ING: 0, X_ED: 0, X_LY: 0, UN_X: 0, IN_X: 0}
        special_fix_tag_prob[tag] = {X_ING: 0, X_ED: 0, X_LY: 0, UN_X: 0, IN_X: 0}

    
    for word in hapax_word_bag:
        fix = 0
        if len(word) < 4:
            continue
        if word[-3] + word[-2] + word[-1] == 'ing':
            fix = X_ING
        elif word[-2] + word[-1] == 'ed':
            fix = X_ED
        elif word[-2] + word[-1] == 'ly':
            fix = X_LY
        elif word[0] + word[1] == 'un':
            fix = UN_X
        elif word[0] + word[1] == 'in':
            fix = IN_X
        tag = hapax_word_bag[word]
        if fix !=0:
            special_fix_tag_count[tag][fix] += 1

    for tag in tag_list:
        tot_hapx_tag_emi = hapax_tag_count[tag]
        for fix in special_fix_list:
            occ = special_fix_tag_count[tag][fix]
            if tot_hapx_tag_emi + lap_coef * p_hapax[tag] * tag_num == 0:
                smoothed_prob = 0
            else:
                smoothed_prob = (occ + lap_coef * p_hapax[tag]) / (tot_hapx_tag_emi + lap_coef * p_hapax[tag] * tag_num)
            if smoothed_prob == 0:
                special_fix_tag_prob[tag][fix] = -math.inf
            else:
                special_fix_tag_prob[tag][fix] = math.log(smoothed_prob)


    for sentence in test:
        count += 1
        word_num = len(sentence)
        vtb_prob_table = np.zeros((tag_num, word_num-2))
        vtb_pointer_table = np.zeros((tag_num, word_num-2))
        word = sentence[1]

        for i in range(tag_num):
            tag = tag_list[i]
            p_ini = init_prob[tag]
            fix = 0
            if word in word_bag:
                if word in emi_hapax_prob[tag]:
                    p_emi = emi_hapax_prob[tag][word]
                else:
                    p_emi = emi_prob[tag][word]
            else:
                if len(word) >= 4:
                    if word[-3] + word[-2] + word[-1] == 'ing':
                            fix = X_ING
                    elif word[-2] + word[-1] == 'ed':
                            fix = X_ED
                    elif word[-2] + word[-1] == 'ly':
                            fix = X_LY
                    elif word[0] + word[1] == 'un':
                            fix = UN_X
                    elif word[0] + word[1] == 'in':
                            fix = IN_X
                if fix !=0 :
                    p_emi = special_fix_tag_prob[tag][fix]
                else:
                    p_emi = emi_prob[tag][UNKNOWN]
            p_tot = p_ini + p_emi
            ptr = i
            vtb_prob_table[i][0] = p_tot
            vtb_pointer_table[i][0] = ptr

        for i in range(2,word_num-1):
            word = sentence[i]
            fix = 0
            for j in range(tag_num):
                tag = tag_list[j]
                if word in word_bag:
                    if word in emi_hapax_prob[tag]:
                        p_emi = emi_hapax_prob[tag][word]
                    else:
                        p_emi = emi_prob[tag][word]
                else:
                    if len(word) >= 4:
                        if word[-3] + word[-2] + word[-1] == 'ing':
                                fix = X_ING
                        elif word[-2] + word[-1] == 'ed':
                                fix = X_ED
                        elif word[-2] + word[-1] == 'ly':
                                fix = X_LY
                        elif word[0] + word[1] == 'un':
                                fix = UN_X
                        elif word[0] + word[1] == 'in':
                                fix = IN_X
                    if fix !=0 :
                        p_emi = special_fix_tag_prob[tag][fix]
                    else:
                        p_emi = emi_prob[tag][UNKNOWN]
                p_tot = float('-inf')
                ptr = 0
                for k in range(tag_num):
                    pre_tag = tag_list[k]
                    p_pair = trans_prob[pre_tag][tag]
                    p_prev = vtb_prob_table[k][i-2]
                    p_tot_temp = p_prev + p_pair + p_emi
                    if p_tot_temp > p_tot:
                        p_tot = p_tot_temp
                        ptr = k
                vtb_prob_table[j][i-1] = p_tot
                vtb_pointer_table[j][i-1] = ptr
        
        path_end = np.argmax(vtb_prob_table[:,-1])
        
        ptr = path_end
        result_tag_list = []
        for i in range(word_num-2, 0, -1):
            tag = tag_list[ptr]
            word = sentence[i]
            result_tag_list.append((word,tag))
            pre_ptr = ptr
            ptr = int(vtb_pointer_table[pre_ptr][i-1])
            
        result_tag_list.reverse()
        result_tag_list.insert(0,("START","START"))
        result_tag_list.append(("END","END"))
        result.append(result_tag_list)
    return result

def training(train_set):
    tag_count = 0
    tag_occ = {}
    emi_occ = {}

    for sentence in train_set:
        tag_count += 1
        for item in sentence:
            word = item[0]
            tag = item[1]
            if tag not in tag_occ:
                tag_occ[tag] = 1
            else:
                tag_occ[tag] += 1
    
    tag_pair_occ = defaultdict(lambda: defaultdict(lambda: 0))
    emi_occ = defaultdict(lambda: defaultdict(lambda: 0))
    word_bag = defaultdict(lambda: defaultdict(lambda: 0))

    count = 0
    for sentence in train_set:
        count += 1
        for i in range(1, len(sentence)): 
            pre_tag = sentence[i-1][1]
            word = sentence[i][0]
            tag = sentence[i][1]   
            tag_pair_occ[pre_tag][tag] += 1
            emi_occ[tag][word] += 1
            word_bag[word][tag] += 1

    return tag_count, word_bag, tag_occ, tag_pair_occ, emi_occ
    

def smoothing(tag_count, word_bag, tag_occ, tag_pair_occ, emi_occ):
    sub_ini_prob = {}
    tag_pair_prob = defaultdict(lambda: defaultdict(lambda: 0))
    emi_prob = defaultdict(lambda: defaultdict(lambda: 0))
    emi_hapax_prob = defaultdict(lambda: defaultdict(lambda: 0))

    hapax_word_bag = {}
    hapax_tag_count = defaultdict(int)
    for word in word_bag:
        if sum(word_bag[word].values()) == 1:
            key = word_bag[word].keys()
            tag = list(key)[0]
            if tag != "START" and key != "END":
                hapax_word_bag[word] = tag

    for word in hapax_word_bag:
        tag = hapax_word_bag[word]
        hapax_tag_count[tag] += 1

    p_hapax = {}
    for key in tag_occ.keys():
        if key != "START" and key != "END":
            p_hapax[key] = 0
    tot_hapax = sum(hapax_tag_count.values())
    for tag in hapax_tag_count:
        p_hapax[tag] = hapax_tag_count[tag]/tot_hapax

    lap_coef = 1e-5
    tag_list = []
    for tag in tag_occ.keys():
        tag_list.append(tag)
    tag_num = len(tag_list)
    for i in range(tag_num):
        tag = tag_list[i]
        if "START" in tag_pair_occ[tag]:
            del tag_pair_occ[tag]["START"]
        if "END" in tag_pair_occ[tag]:
            del tag_pair_occ[tag]["END"]
        
    sub_ini_occ = {}
    for key in tag_occ:
        if key not in tag_pair_occ["START"]:
            sub_ini_occ[key] = 0
        else:
            sub_ini_occ[key] = tag_pair_occ["START"][key]

    count = 0
    for key in sub_ini_occ:
        if key == "START" or key == "END":
            continue
        count += 1
        ini_occ = sub_ini_occ[key]
        smoothed_prob = (ini_occ + lap_coef) / (tag_count + lap_coef * tag_num)
        log_prob = math.log(smoothed_prob)
        sub_ini_prob[key] = log_prob

    count = 0
    for key_1 in tag_pair_occ:
        if key == "START" or key == "END":
            continue
        count += 1
        tot_pair = sum(tag_pair_occ[key_1].values())
        for key_2 in tag_occ:
            if key == "START" or key == "END":
                continue
            if key_2 not in tag_pair_occ[key_1]:
                occ = 0
            else:
                occ = tag_pair_occ[key_1][key_2]
            
            smoothed_prob = (occ + lap_coef) / (tot_pair + lap_coef * tag_num)
            log_prob = math.log(smoothed_prob)
            tag_pair_prob[key_1][key_2] = log_prob

    count = 0
    for key_1 in emi_occ:
        if key_1 == "START" or key_1 == "END":
            continue
        count += 1
        tot_emi = sum(emi_occ[key_1].values())
        for word in word_bag:
            if word not in emi_occ[key_1]:
                occ = 0
            else:
                occ = emi_occ[key_1][word]
            if occ ==1:
                smoothed_prob = (occ + lap_coef * p_hapax[key_1]) / (tot_emi + lap_coef * p_hapax[key_1] * tag_num)
                log_prob = math.log(smoothed_prob)
                emi_hapax_prob[key_1][word] = log_prob
            else:
                smoothed_prob = (occ + lap_coef * p_hapax[key_1]) / (tot_emi + lap_coef * p_hapax[key_1] * tag_num)
                if smoothed_prob == 0:
                    log_prob = float('-inf')
                else:
                    log_prob = math.log(smoothed_prob)
                emi_prob[key_1][word] = log_prob
                smoothed_prob = (0 + lap_coef * p_hapax[key_1]) / (tot_emi + lap_coef * p_hapax[key_1] * tag_num)
                if smoothed_prob == 0:
                    emi_prob[key_1]["UNKNOWN"] = float('-inf')
                else:
                    emi_prob[key_1]["UNKNOWN"] = math.log((0 + lap_coef * p_hapax[key_1]) / (tot_emi + lap_coef * p_hapax[key_1] * tag_num))

    return sub_ini_prob, tag_pair_prob, emi_prob, emi_hapax_prob 
