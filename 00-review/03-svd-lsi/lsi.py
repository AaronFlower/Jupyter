# encoding: utf-8

import numpy as np

def raw_count_tf(wordSet, docWords):
    '''
        计算 term-frequency, 词语出现的相对频率
        BoW: Bag of Words, 词袋模型
    '''
    tfBow = dict.fromkeys(wordSet, 0)
    for word in docWords:
        tfBow[word] += 1
    return list(tfBow.values())

def binary_df(tfBow):
    '''
        计算单词的 docuemnt-frequency, 单词在所有文档中出现的频率。
        在我们的例子中的词典的单词至少会出现在一个文档中。
    '''
    occur = tfBow >= 1 # fancy indexing
    tfBow[occur] = 1
    tfBow[~occur] = 0
    return tfBow

class LatentSemanticIndex(object):
    '''
        Latent Semantic Indexing
    '''
    def __init__(self,
                 calc_tf = raw_count_tf,
                 calc_df = binary_df,
                 k = 2):
        '''
            Initialization
            - calc_tf   : compute term frequency in a document. (local)
            - calc_df   : compute term in all documents frequnecy.(global)
            - k         : the top k eigenvectors to use.
        '''
        self.calc_tf = calc_tf
        self.calc_df = calc_df
        self.k = k
        self.wordSet = set()
        self.U, self.Sigma, self.VT = None, None, None
        return

    def gen_docs(self, corpus):
        '''
            将语料库中和每一个文档解析成单独的单词，并把所有单词入在一个 Set 中。
            在这个例子中我们的每一个文档都只是一句话而已，所以处理起来十分简单。
        '''
        docs = []
        wordSet = set()
        for doc in corpus:
            words = doc.split(' ')
            words = [word.lower() for word in words]
            wordSet = wordSet | set(words)
            docs.append(words)
        return docs, wordSet

    def trainLsi(self, corpus):
        '''
            Generate LIS according to corpus.
        '''
        docs, wordSet = self.gen_docs(corpus)
        self.wordSet = wordSet

        m, n = len(wordSet), len(docs)
        A = np.zeros((m, n))

        for i, doc in enumerate(docs):
            A[:, i] = self.calc_tf(wordSet, doc)

        for i  in range(m):
            A[i] = A[i] * self.calc_df(A[i].copy())

        self.U, Sigma, self.VT = np.linalg.svd(A)
        self.Sigma = np.eye(len(Sigma)) * Sigma

        return docs, wordSet, A

def run():
    corpus = [
        'Shipment of gold damaged in a fire',
        'Delivery of silver arrived in a silver truck',
        'Shipment of gold arrived in a truck'
    ]

    # query = 'gold silver truck'

    lis = LatentSemanticIndex();
    docs, wordSet, A = lis.trainLsi(corpus)
    return docs, wordSet, A
