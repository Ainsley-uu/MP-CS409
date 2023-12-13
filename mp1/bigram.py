def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    # Keep this in the provided template
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)


    max_vocab_size = None

    # unigram
    pos_word_map, neg_word_map = create_word_maps(train_set, train_labels, n = 1, max_size = max_vocab_size)
    num_pos_words = sum(pos_word_map.values())
    num_neg_words = sum(neg_word_map.values())

    pos_norm = num_pos_words + unigram_laplace * (len(pos_word_map) + 1)
    neg_norm = num_neg_words + unigram_laplace * (len(neg_word_map) + 1)

    # bigram
    pos_word_map_bi, neg_word_map_bi = create_word_maps(train_set, train_labels, n = 2, max_size = max_vocab_size)
    num_pos_words_bi = sum(pos_word_map_bi.values())
    num_neg_words_bi = sum(neg_word_map_bi.values())

    pos_norm_bi = num_pos_words_bi + bigram_laplace * (len(pos_word_map_bi) + 1)
    neg_norm_bi = num_neg_words_bi + bigram_laplace * (len(neg_word_map_bi) + 1)


    yhats = []
    for x in tqdm(dev_set,disable=silently):
        # unigram probabilities
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1 - pos_prior)
        for w in x:
            if w not in pos_word_map:
                # unk
                pos_prob += math.log(unigram_laplace / pos_norm)
            else:
                pos_prob += math.log((pos_word_map[w] + unigram_laplace) / pos_norm)

            if w not in neg_word_map:
                # unk
                neg_prob += math.log(unigram_laplace / neg_norm)
            else:
                neg_prob += math.log((neg_word_map[w] + unigram_laplace) / neg_norm)


        # bigram probabilities
        pos_prob_bi = math.log(pos_prior)
        neg_prob_bi = math.log(1 - pos_prior)

        for i in range(len(x) - 1):
            w = " ".join(x[i: i + 2])

            if w not in pos_word_map_bi:
                # unk
                pos_prob_bi += math.log(bigram_laplace / pos_norm_bi)
            else:
                pos_prob_bi += math.log((pos_word_map_bi[w] + bigram_laplace) / pos_norm_bi)

            if w not in neg_word_map_bi:
                # unk
                neg_prob_bi += math.log(bigram_laplace / neg_norm_bi)
            else:
                neg_prob_bi += math.log((neg_word_map_bi[w] + bigram_laplace) / neg_norm_bi)


        pos_prob_total = (1 - bigram_lambda) * pos_prob + bigram_lambda * pos_prob_bi
        neg_prob_total = (1 - bigram_lambda) * neg_prob + bigram_lambda * neg_prob_bi

        if pos_prob_total > neg_prob_total:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats