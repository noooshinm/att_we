import numpy as np
import random
import os
from collections import Counter
import zipfile
import tensorflow as tf


#load wordmap
def load_wordmap(wordmap):
    id2word = {}
    with open(wordmap) as f:
        f.readline()
        for l in f:
            word, id = l.strip().split()
            id = int(id)
            id2word[id] = word
    return id2word

def read_data(id2word, wTopicAsg_file, dir, word_fname, topic_fname):
    words = []
    topics = []

    word_path = dir + word_fname
    word_file = open(word_path, 'w')
    topic_path = dir + topic_fname
    topic_file = open(topic_path, 'w')

    with open(wTopicAsg_file) as f:
        lines = f.readlines()
        for line in lines:
            doc = line.strip().split()
            for wt in doc:
                w, t = wt.strip().split(':')
                w_idx = int(w)
                t_idx = int(t)
                words.append(w_idx)
                topics.append(t_idx)

                word = id2word[w_idx]
                word_file.write(word+' ')
                topic_file.write(t_idx)
            word_file.write('\n')
            topic_file.write('\n')

            words.append('<eos>')
            topics.append('<eos>')
    word_file.close()
    topic_file.close()

    return words, topics


def batch_gen(batch_size, win_size, words, n_neg, isTopical, topics=None):

    while True:
        mem_size = 2*win_size
        target = np.zeros([batch_size])
        #labels = np.zeros([batch_size, vocab_size])
        context = np.zeros([batch_size, mem_size])
        noise_words = np.zeros([batch_size, n_neg])

        if isTopical:
            target_topic = np.zeros([batch_size])
            #labels_topic = np.ndarray([batch_size, n_topic])
            context_topic = np.zeros([batch_size, mem_size])
            noise_topics = np.zeros([batch_size, n_neg])


        for i in range(batch_size):
            m = random.randrange(win_size, len(words)-win_size)
            #labels[i][words[m]] = 1 or labels = tf.one_hot(target, vocab_size)
            target[i] = words[m]
            context[i] = np.concatenate((words[m-win_size:m], words[m+1:m+win_size+1]))
            negs = []


            while len(negs)< n_neg:
                neg_id = random.randrange(0, len(words)) #instead make a table of all vocab words and their frequncy like gensim and draw noise samples from that
                if words[neg_id]!=words[m]:
                    negs.append(words[neg_id])
            noise_words[i] = negs

            if isTopical:
                #labels_topic[i][topics[m]]= 1
                target_topic[i] = topics[m]
                context_topic[i] = np.concatenate((topics[m-win_size:m], topics[m+1:m+win_size+1]))
                negs_topics = []

                while len(negs_topics)<= n_neg:
                    neg_id_topic = random.randrange(0,len(topics))
                    if topics[neg_id_topic] != topics[m]:
                        negs_topics.append(topics[neg_id_topic])
                noise_topics[i] = negs_topics


        if isTopical:
            yield target, context, noise_words, target_topic, context_topic, noise_topics
        else:
            yield target, context, noise_words



def batch_gen_test(batch_size, win_size, words, n_neg, isTopical, topics=None):

    while True:
        mem_size = 2*win_size
        target = np.zeros([batch_size])
        #labels = np.zeros([batch_size, vocab_size])
        context = np.zeros([batch_size, mem_size])
        noise_words = np.zeros([batch_size, n_neg])

        if isTopical:
            target_topic = np.zeros([batch_size])
            #labels_topic = np.ndarray([batch_size, n_topic])
            context_topic = np.zeros([batch_size, mem_size])
            noise_topics = np.zeros([batch_size, n_neg])

        m = mem_size
        for i in range(batch_size):
            target[i] = words[m]
            context[i] = np.concatenate((words[m-win_size:m], words[m+1:m+win_size+1]))
            negs = []

            while len(negs)< n_neg:
                neg_id = random.randrange(0, len(words)) #instead make a table of all vocab words and their frequncy like gensim and draw noise samples from that
                if words[neg_id]!=words[m]:
                    negs.append(words[neg_id])
            noise_words[i] = negs

            if isTopical:
                #labels_topic[i][topics[m]]= 1
                target_topic[i] = topics[m]
                context_topic[i] = np.concatenate((topics[m-win_size:m], topics[m+1:m+win_size+1]))
                negs_topics = []

                while len(negs_topics)<= n_neg:
                    neg_id_topic = random.randrange(0,len(topics))
                    if topics[neg_id_topic] != topics[m]:
                        negs_topics.append(topics[neg_id_topic])
                noise_topics[i] = negs_topics

            m += 1
            if m >= len(words):
                m = mem_size

        if isTopical:
            yield target, context, noise_words, target_topic, context_topic, noise_topics
        else:
            yield target, context, noise_words


def read_data1(fname):
    count = []
    word2idx = {}
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise("[!] Data %s not found" % fname)

    words = []
    for line in lines:
        words.extend(line.split())

    if len(count) == 0:
        count.append(['<eos>', 0])

    count[0][1] += len(lines)
    count.extend(Counter(words).most_common())

    if len(word2idx) == 0:
        word2idx['<eos>'] = 0

    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    data = list()
    for line in lines:
        for word in line.split():
            index = word2idx[word]
            data.append(index)
        data.append(word2idx['<eos>'])

    print("Read %s words from %s" % (len(data), fname))
    return data, len(word2idx), word2idx


def read_analogy(analogy_file, word2id):
    '''
    :param analogy_file:
    :param word2id:
    :return: array of analogy questions in which each row contains the vocab_ids of 4 words a,b,c,d
            in each question [N,4]
    '''
    analogy_q = []
    q_missed = 0
    with open(analogy_file) as f:
        for line in f:
            if line.startswith(':'):
                continue
            words = line.strip().lower().split()
            w_ids = [word2id.get(w.strip()) for w in words]
            if None in w_ids or len(w_ids) != 4:
                q_missed+=1
            else:
                analogy_q.append(np.array(w_ids))
    print ("Read %s analogy question from %s" % (len(analogy_q), analogy_file))
    print ("skipped %s analogy questions" % q_missed)
    analogy_questions = np.array(analogy_q, dtype=np.int32)
    return analogy_questions


#step2: read data input, convert it to string and tokenize the whole corpus into words
def load_text8 (file_path):
    with zipfile.ZipFile(file_path) as f:
        input_string = tf.compat.as_str(f.read(f.namelist()[0]))
        word_tokens = input_string.split()
    return word_tokens

#step3: create vocabulary
def read_text8(words):
    count = []
    word2idx = {}

    count.extend(Counter(words).most_common())

    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    data = list()

    for word in words:
        index = word2idx[word]
        data.append(index)

    n_words = len(words)
    print ('total number of words: ', n_words)
    n_train = int(n_words * 0.9)
    train_set = data[0:n_train]
    valid_set = data[n_train: n_words]
    print("Read %s total words" % (len(data)))
    print ('Read %s train words' % (len(train_set)))
    print ('Read %s valid words' % (len(valid_set)))
    return train_set, valid_set, len(word2idx), word2idx


# def sth():
#     text8_filePath = './attWE_dataset/text8.zip'
#     tokens = load_text8(text8_filePath)
#     train_words, valid_words, vocabSize, vocab = read_text8(tokens)
#
#     analogy_questions  = read_analogy('./attWE_dataset/questions-words.txt', vocab)
#     h = analogy_questions[:, 0]
#     print (h.shape)
#
# sth()

# print (missed[0])
# words = ['athens greece baghdad iraq']
# w_ids = [vocab.get(w.strip()) for w in words]


# import itertools
#
# x2 = list(itertools.islice(vocab.items(), 0, 4))
#
#
# id = vocab.get('athens')
# id2 = vocab['greece']
# id3 = vocab['baghdad']
# id4 = vocab['iraq']
# print (id4)


# train_data = read_data1('/home/nooshin/PycharmProjects_Datasets/attWE_dataset/ptb.train.txt')
# print (len(train_data))
# print (type(train_data))
# print (train_data[0:3])







