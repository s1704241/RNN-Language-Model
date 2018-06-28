import numpy as np
import pandas as pd
from utils import *

data_folder = '../data'
vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0,
                       names=['count', 'freq'], )

# vocab_new = vocab1.sort_values('count', ascending=False)
num_to_word1 = dict(enumerate(vocab.index[:2000]))
word_to_num1 = invert_dict(num_to_word1)

skip_embeddings_index = {}
cbow_embeddings_index = {}
glove_embeddings_index = {}
f = open('100d_SKIP.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    skip_embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(skip_embeddings_index))

f = open('100d_CBOW.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    cbow_embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(cbow_embeddings_index))

f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(glove_embeddings_index))

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def softmax(x):
	xt = np.exp(x - max(x))
	return xt / sum(xt)

def grad(x):
	return x*(1-x)

def make_onehot(i, n):
	y = np.zeros(n)
	y[i] = 1
	return y

def make_SKIP(i,n):
    '''
    n is the dimension of vector space, i is the word index in dataset
    '''

    try:
        word = num_to_word1[i]
        embedding = skip_embeddings_index[word]
    except KeyError:
        word = 'unknown'
        embedding = skip_embeddings_index[word]

    return embedding

def make_Glove(i,n):
    try:
        word = num_to_word1[i]
        embedding = glove_embeddings_index[word]
    except KeyError:
        word = 'unknown'
        embedding = glove_embeddings_index[word]

    return embedding

def make_CBOW(i,n):
    try:
        word = num_to_word1[i]
        embedding = cbow_embeddings_index[word]
    except KeyError:
        word = 'unknown'
        embedding = cbow_embeddings_index[word]

    return embedding







def fraq_loss(vocab, word_to_num, vocabsize):
	fraction_lost = float(sum([vocab['count'][word] for word in vocab.index if (not word in word_to_num) and (not word == "UNK")]))
	fraction_lost /= sum([vocab['count'][word] for word in vocab.index if (not word == "UNK")])
	return fraction_lost

def adjust_loss(loss, fracloss, q, mode='basic'):
	if mode == 'basic':
		# remove freebies only: score if had no UNK
		return (loss + fracloss*np.log(fracloss))/(1 - fracloss)
	else:
		# remove freebies, replace with best prediction on remaining
		return loss + fracloss*np.log(fracloss) - fracloss*np.log(q)



class MultinomialSampler(object):
    """
    Fast (O(log n)) sampling from a discrete probability
    distribution, with O(n) set-up time.
    """

    def __init__(self, p, verbose=False):
        n = len(p)
        p = p.astype(float) / sum(p)
        self._cdf = np.cumsum(p)

    def sample(self, k=1):
        rs = np.random.random(k)
        # binary search to get indices
        return np.searchsorted(self._cdf, rs)

    def __call__(self, **kwargs):
        return self.sample(**kwargs)

    def reconstruct_p(self):
        """
        Return the original probability vector.
        Helpful for debugging.
        """
        n = len(self._cdf)
        p = np.zeros(n)
        p[0] = self._cdf[0]
        p[1:] = (self._cdf[1:] - self._cdf[:-1])
        return p

def multinomial_sample(p):
    """
    Wrapper to generate a single sample,
    using the above class.
    """
    return MultinomialSampler(p).sample(1)[0]
