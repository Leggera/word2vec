import numpy as np
import random
import nltk

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad
from nltk.tokenize import PunktSentenceTokenizer



def tokenize_to_sentences(article):
    train_text = open (article, 'r')
    sample_text = open (article, 'r')
    custom_sent_tokenizer = PunktSentenceTokenizer(train_text.read().decode('utf-8'))
    tokenized = custom_sent_tokenizer.tokenize(sample_text.read().decode('utf-8'))
    return tokenized

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    #return (x.T / np.sqrt(np.sum((x.T)**2, axis=0))).TZ
    s = np.sqrt(np.sum(x * x, axis=1).reshape(-1, 1))
    if (s.all() > 0):
        return x / s
    else:
        return x

    ### END YOUR CODE
    
def negSamplingCostAndGradient_old(predicted, target, outputVectors, dataset, neg_idx, prev_idx,
    K=10):
    
    negative_set = set(neg_idx)
    negative_set.discard(target)
    negative_idx = list(negative_set)
    negative_set = negative_set.difference(prev_idx)
    negative_idx = list(negative_set)
    s = outputVectors.shape
    if not(negative_idx):
        Z = np.zeros((s))
        return None
        return 0, 0, Z, Z.T, [], Z.T, Z.T
    prev_idx += negative_idx
    
    '''print "old"
    print negative_idx
    print target'''
    target_vector = outputVectors[target]
    U = outputVectors[negative_idx]
    grad = np.zeros(outputVectors.shape)

    prod_t = np.dot(target_vector, predicted)
    sig = sigmoid(prod_t)
    g = sig - 1
    sig_n = sigmoid(np.dot(U, predicted))

    
    grad[target] =  g * predicted 
    h =  sig_n * U.T
    n =  -np.log(1 - sig_n)
    
    gradPred = g * target_vector.T + np.sum(h, axis=1)

    cost = -np.log(sig) + np.sum(n)

    nn =  sig_n[:, np.newaxis] * predicted
    k = 0
    for _idx in negative_idx:
        #cost += n[g]
        #gradPred += h[:, g]
        grad[_idx] += nn[k]
        print nn[k]/50
        k  =  k + 1
    return cost, gradPred, grad

def skipgram_old(currentWord, C, contextWords, tokens, inputVectors, outputVectors, neg_idx, prev_idx,
    dataset, word2vecCostAndGradient = negSamplingCostAndGradient_old):
    print "#################################################################"
    print prev_idx
    print "#################################################################"
    contextWords_set = set(contextWords)
    contextWords_set.discard(currentWord)
    contextWords = list(contextWords_set)

    gradIn = np.zeros(inputVectors.shape)
    t = tokens[currentWord]
    predicted = inputVectors[t]
    p_list = map((lambda x: word2vecCostAndGradient(predicted, tokens[x], outputVectors, dataset, neg_idx, prev_idx)), contextWords)
    cost, gradIn[t], gradOut = map(sum, zip(*p_list))
    return cost, gradIn, gradOut

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, prev_idx,
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    K = 10
    neg_idx = [dataset.sampleTokenIdx() for k in range(K)]
    negative_set = set(neg_idx)
    negative_set.discard(target)
    negative_set = negative_set.difference(prev_idx)
    negative_idx = list(negative_set)
    
    s = outputVectors.shape[1]
    if not(negative_idx):
        Z = np.zeros((s, 1))
        return None
        return 0, 0, Z, Z.T, [], Z.T, Z.T, []
    prev_idx += negative_idx
    '''print "new"
    print negative_idx
    print target'''
    #print target
    #print negative_idx
    target_vector = outputVectors[target]
    U = outputVectors[negative_idx]
    negative_idx.append(target)
    

    #grad = np.zeros(outputVectors[negative_idx].shape)# 10^6 x 100 #TODO
    negative_idx.remove(target)#TODO

    prod_t = np.dot(target_vector, predicted)
    sig = sigmoid(prod_t)
    g = sig - 1
    sig_n = sigmoid(np.dot(U, predicted))

    
    #grad[target] =  g * predicted 
    h =  sig_n * U.T
    n =  -np.log(1 - sig_n)
    
    #gradPred =  np.sum(h, axis=1)

    #cost = -np.log(sig) + np.sum(n)#TODO cost is only needed for checking
    #print np.log(sig)
    #exit()
    #print n
    cost = list(n)
    #cost.append(list(n))
    #cost.append()
    
    #cost = np.concatenate((n, np.array([-np.log(sig)])))
    grad =  sig_n[:, np.newaxis] * predicted
    #k = 0
    #for _idx in negative_idx:
        #cost += n[g]
        #gradPred += h[:, g]
    #print nn.shape
    #print (g * predicted).reshape(1, -1).shape
    
    
    #print grad
    #exit()
        #k  =  k + 1
    ### END YOUR CODE
    
    return cost, -np.log(sig), h, g * target_vector.T, negative_idx, grad, g * predicted, [target]


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, prev_idx,
    dataset, word2vecCostAndGradient = negSamplingCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE

    #gradIn = np.zeros(inputVectors.shape)#TODO
    #gradOut = np.zeros(np.shape(outputVectors))
    #print inputVectors.shape
    t = tokens[currentWord]
    predicted = inputVectors[t]
    #print contextWords
    contextWords_set = set(contextWords)
    contextWords_set.discard(currentWord)
    contextWords = list(contextWords_set)
    #print contextWords
    #exit()
    p_list = map((lambda x: word2vecCostAndGradient(predicted, tokens[x], outputVectors, dataset, prev_idx)), contextWords)
    p_list = [p for p in p_list if p != None]
    #map((lambda x: if x[3]))
    cost = 0
    #for p in p_list:
        #cost += p[0]
        #gradIn[t] += p[1]
        #neg_idx = p[2]
        #gradOut[neg_idx] += p[3]
    k = 0
    cost_stack = np.array([])
    cost_target = 0
    gradIn_stack = np.array([])
    arr_neg_idx = np.array([])
    gradOut_stack = np.array([])
    gradIn_target_stack = np.array([])
    gradOut_target_stack = np.array([])
    target_indices = []
    for x in zip(*p_list):
        #print x#cost
        if (k == 0):
            #cost = sum(x)
            #vstack
            list_x = []
            for w in x:
                if (w):
                    list_x += w
            cost_stack = np.asarray(list_x)
        if (k == 1):
            cost_target = sum(x) 
        if (k == 2):
            #print
            #gradIn[t] = sum(x)
            #vstack 
            gradIn_stack = np.concatenate([j for j in x if np.sum(j)], axis = 1)
            #print gradIn_stack
        if (k == 3):
            gradIn_target_stack = np.vstack([j for j in x if np.sum(j)])
        if (k == 4):
            #asarray
            list_x = []
            #print x
            for w in x:
                if (w):
                    list_x += w
            arr_neg_idx = np.asarray(list_x)
        if (k == 5):
            '''print negative_idx
            print gradOut.shape
            print x'''
            #for idx, x1 in zip(negative_idx, x):
                #gradOut[idx] += x1
            gradOut_stack = np.vstack([j for j in x if np.sum(j)])
            #print gradOut_stack.shape
            #print gradOut_stack
            #exit()
            #vstack
        if (k == 6):
            gradOut_target_stack = np.vstack([j for j in x if np.sum(j)])
        if (k == 7):
            for w in x:
                if (w):
                    target_indices += w
            break
        k += 1
    
    '''i = 0
    for w in contextWords:
        print tokens[w]
    for idx in negative_idx:
        #for x in idx:
        i += len(idx) - 1
        #print "idx", idx
        #print "i", i
        print "last_id", idx[-1]
        print gradOut_stack[i]
        i += 1
    exit()'''
    #map((lambda x: print x), zip(*p_list))  
    #exit()
    #cost, gradIn[t], gradOut, h = map((lambda x: sum(x[0])), zip(*p_list))#unzipped list

    ### END YOUR CODE
    
    return cost_stack, cost_target, target_indices, gradIn_stack, gradIn_target_stack, arr_neg_idx, gradOut_stack, gradOut_target_stack

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, it, word2vecCostAndGradient = negSamplingCostAndGradient):   
    prev_idx = []
    batchsize = 50
    cost = 0.0
    #cost_old = 0.0
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    #grad = np.zeros(wordVectors.shape)
    #grad_old = np.zeros(wordVectors.shape)
    count = 0
    first = True
    I_in = np.array([])
    GradIn = np.array([])
    I_out = np.array([])
    GradOut = np.array([])
    GradOut_target = np.array([])
    T = []
    for i in it:
        C1 = random.randint(1, C)
        
        denom = 1
        #print i
        centerword, context = i
        #print(centerword)
        #print batchsize
        t = tokens[centerword]
        c, c_target, idx_in, gin, gin_target, idx_out, gout, gout_target = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, prev_idx, dataset, word2vecCostAndGradient)
        #print idx_out
        count += 1
        if (count < batchsize):
            if ((len(idx_in)) and (len(idx_out))):
                '''h = [i + N/2 for i in idx_out]
                grad[h, :] += gout / batchsize
                h = [i + N/2 for i in idx_in]
                grad[h, :] += gout_target / batchsize
                grad[t, :] += (np.sum(gin, axis=1) + np.sum(gin_target, axis = 0)) / batchsize'''
                #print count
                #print "OK"
                cost += (sum(c) + c_target) / batchsize / denom
                if (first):
                    first = False
                    I_in = idx_in
                    I_out = idx_out
                    GradIn = (np.sum(gin, axis=1) + np.sum(gin_target, axis = 0)).reshape(-1, 1) / batchsize
                    GradOut = gout
                    GradOut_target = gout_target
                    T = [t]
                else:
                    I_in = np.concatenate([I_in, idx_in])
                    I_out = np.concatenate([I_out, idx_out])
                    
                    try:
                        GradIn = np.concatenate([GradIn, (np.sum(gin, axis=1) + np.sum(gin_target, axis = 0)).reshape(-1, 1) / batchsize], axis = 1)
                    except:
                        print idx_in
                        print idx_out
                        print centerword
                        print gin.shape
                        print gin_target.shape
                        print GradIn.shape
                        exit()
                    GradOut = np.concatenate([GradOut, gout])
                    GradOut_target = np.concatenate([GradOut_target, gout_target])
                    T += [t]
                
                '''grad[T, :] += GradIn.T
                h = [i + N/2 for i in I_out]
                grad[h, :] += GradOut / batchsize
                h = [i + N/2 for i in I_in]
                grad[h, :] += GradOut_target / batchsize'''
                #print "WHAT"
            else:
                continue
        else:
            return cost, I_in, GradIn, I_out, GradOut, GradOut_target, T   
            #return cost, grad
    return cost, I_in, GradIn, I_out, GradOut, GradOut_target, T
    #return cost, grad
def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        while True:
            yield tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
                for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext
    C = 10
    word = dataset.getRandomContext(C)#can input random(C) inside this function
    it = iter(word)
    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, C, it, negSamplingCostAndGradient), dummy_vectors)
if __name__ == "__main__":

    test_word2vec()
