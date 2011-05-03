"""
Created on Jan 31, 2010
@author: nrolland
From Gregor Heinrich's most excellent text 'Parameter estimation for text analysis'
"""

"""
Adapted for Author-Recipient-Topic model on Apr 27, 2011
@author: askory, gparent
From McCallum et al., The Author-Recipient-Topic Model for Topic and Role Discovery in Social Networks, with Application to Enron and Academic Email
"""

import sys, getopt
import collections, math, array, numpy
from numpy import zeros, ones
from collections import defaultdict
from scipy.special import gamma, gammaln
import random, operator, simplejson
#This code assumes mongo models with the documents in them
from models import SparsePost, TestSparsePost
from stops import STOPS
from copy import deepcopy

# We test if numpy is installed
np = 'numpy' in sys.modules

verbose =False
verbose_read = False
n_verbose_iterate  = 50

def ismultiple(i, n):
    return i - (i/n)*n == 0
        
def indice(a):
    if np:
        i =numpy.argmax(a)
    else:        
        for i,val in enumerate(a):
            if val > 0:
                break
    return int(i)

def righshift(ar, pos = 0):
    ar[pos + 1:] = ar[pos:-1]
    ar[pos] = float('-Infinity')
    return ar
 
def indicenbiggest(ar,n):
    ln = min(abs(n),len(ar))
    n_biggest = [float('-Infinity')] *ln
    n_biggest_index = [0] *ln
    
    for i, val in enumerate(ar):
        for i_biggest in xrange(n):
            if val > n_biggest[i_biggest]:
                n_biggest = righshift(n_biggest, i_biggest)
                n_biggest[i_biggest] = val
                n_biggest_index = righshift(n_biggest_index, i_biggest)
                n_biggest_index[i_biggest] = i
                break
 
    return n_biggest_index

def flatten(x):
    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result
 
def mat(shape, val=0):
    if np:
        ret = numpy.zeros(shape)
    else:
        ret = [val]*shape[-1]
        for n_for_dim in reversed(shape[:-1]):
            tret = []
            for i in range(n_for_dim):
                tret.append(ret[:])
            ret= tret
            
    return ret

def oneinrow(ar, row_id):
    """Returns array ar where all element in
    row_id have been replaced with ones"""
    
    tar = ar[:]
    tar[row_id] = numpy.ones(len(ar[row_id]))
    return tar

def oneincol(ar, col_id):
    """Returns array ar where all element in
    col_id have been replaced with ones"""

    tar = ar[:]
    tar[:,col_id] = numpy.ones(len(ar[:,col_id]))
    return tar

def logdelta(v):
    """Returns the difference between the sum of log(f(x))
    and log(sum of f(x)) where f is the gamma function"""
    sigma = 0
    sigmagammaln = 0
    for i, x_i in enumerate(v):
        sigma        += x_i
        sigmagammaln +=  gammaln(x_i)
    return sigmagammaln - gammaln(sigma)

def normalize(ar):
    """normalizes an array"""
    if np:
        ar /= numpy.sum(ar)
    else:
        s = sum(ar)
        ar = [ar[i] / s for i in range(len(ar))]
    return ar


def multinomial(n_add, param, n_dim = 1, normalizeit = True):
    """
    n_add : number of samples to be added for each draw
    param : param of multinomial law
    n_dim : number of samples
    """
    if normalizeit:
        param = normalize(param)
    if np:
        res = numpy.random.multinomial(n_add, param, n_dim)           
    else:
        # if we don't have numpy, run the multinomial sampling ourselves
        res = []
        cdf = [sum(param[:1+end]) for end in range(len(param))]
        zerosample = [0]*len(param)
        for i_dim in range(n_dim):
            sample = zerosample[:]
            for i_add in range(n_add):
                r = random.random()
                for i_cdf, val_cdf in enumerate(cdf):
                    if r < val_cdf : break
                sample[i_cdf] += 1
            res.append(sample)
   
    if n_dim == 1:
        res = res[0]
    return res
        
class SparseDocCollection(list):
    def __init__(self,docs):
        super(SparseDocCollection,self).__init__(docs)
        self.vocabulary = list(set([w for d in docs for w in d.vocabulary]))

class GraphModel(object):
    """Base class for LDA and ART models"""
    def __init__(self,ndocs=-1,ntopics=10,doc_model=SparsePost):
        self.tsavemostlikelywords = 30
        if ndocs == -1:
            self.docs = SparseDocCollection(SparsePost.objects())
            self.ndocs = len(self.docs)
        else:
            self.ndocs = ndocs
            self.docs = SparseDocCollection(random.sample(doc_model.objects(),self.ndocs))
        self.ntopics = ntopics
        self.niters = 100

        #Both LDA and ART have these count variables
        self.nterm_by_topic_term = [defaultdict(int) for topic in xrange(self.ntopics)]
        self.nterm_by_topic      = zeros((self.ntopics, ))



    def info(self):
        print "# of documents : ", len(self.docs)
        print "# of terms  : ", len(self.docs.vocabulary)
        print "# of words  : ", sum(map(lambda doc:doc.n_words, self.docs))
        print "# of topics:  ", self.ntopics

    def printit(self):
        for topic in range(self.ntopics):
            items = self.nterm_by_topic_term[topic].items()
            items.sort(lambda x,y: cmp(y[1],x[1]))
            print "MF word for topic", topic, ":",
            i = 0
            for word in items:
                if word[0] not in STOPS:
                    print word[0]+",",
                    i += 1
                    if i == self.tsavemostlikelywords:
                        print "\n"
                        break
        
class LDAModel(GraphModel):
    def __init__(self,*args):
        super(LDAModel,self).__init__(*args)
        #prior among topic in docs
        self.falpha = 0.5
        #prior among words in topics
        self.fbeta = 0.5

        #initializing the count variables
        self.ntopic_by_doc_topic = zeros((len(self.docs), self.ntopics))
        self.ntopic_by_doc       = zeros((len(self.docs), ))
        self.z_topic          = [zeros((doc.n_words, )) for doc in self.docs]
 
        
    def add(self,doc_id, term_id, topic, qtty=1.):
        self.ntopic_by_doc_topic[doc_id][topic] += qtty
        self.ntopic_by_doc      [doc_id]   += qtty             
        self.nterm_by_topic_term[int(topic)][term_id] += qtty
        self.nterm_by_topic     [topic] += qtty
        
    def remove(self, doc_id, term_id, topic):
        self.add(doc_id, term_id, topic, -1.)
    
    def loglikelihood(self):
        """Not quite convinced that this really computes log likelihood..."""
        loglike = 0
        for k in xrange(self.ntopics):
            loglike += logdelta(map(operator.add, self.nterm_by_topic_term[k].values(), self.beta.values()))
        loglike -= logdelta(self.beta.values()) * self.ntopics

        for m in xrange(len(self.docs)):
            loglike += logdelta(map(operator.add, self.ntopic_by_doc_topic[m][:], self.alpha))
        loglike -= logdelta(self.alpha) * len(self.docs)
        return loglike
        
        
    def initialize(self):
        """initialize the hidden assignments"""
        self.alpha = [self.falpha / self.ntopics] * self.ntopics
        self.beta  = defaultdict(lambda: self.fbeta / len(self.docs.vocabulary))
        model_init = [1. / self.ntopics] * self.ntopics
        print "initial seed"
        # we randomly assign a topic to every word in every document
        # todo: do the same for recipient
        for doc_id, doc in enumerate(self.docs):
            topic_for_words = multinomial(1, model_init,doc.n_words)
            i_word = 0
            for term_id in doc.word_counts:
                for i_term_occ in xrange(doc.word_counts[term_id]):
                    i_topic =  indice(topic_for_words[i_word])
                    self.z_topic[doc_id][i_word] = i_topic
                    self.add(doc_id, term_id,i_topic)
                    i_word += 1     

    def iterate(self):
        """one iteration of gibbs sampling"""
        for doc_id, doc in enumerate(self.docs):
            i_word =0 
            for term_id in doc.word_counts:
                for i_term_occ in xrange(doc.word_counts[term_id]):
                    self.remove(doc_id, term_id, self.z_topic[doc_id][i_word])
                    param = [(self.nterm_by_topic_term[topic][term_id] + self.beta[term_id]) / ( self.nterm_by_topic[topic] + self.fbeta) * \
                             (self.ntopic_by_doc_topic[doc_id][topic] +  self.alpha[topic] ) / ( self.ntopic_by_doc[doc_id] + self.falpha) for topic in range(self.ntopics)]
                    new_topic = indice(multinomial(1, param))
                    self.z_topic[doc_id][i_word] = new_topic
                    self.add(doc_id, term_id, new_topic)
                    i_word += 1
            if n_verbose_iterate > -1 and ismultiple(doc_id, n_verbose_iterate):
                print " doc : ", doc_id ,"/" , len(self.docs)

    def run(self,niters,burnin = 100):
        old_lik = -999999999999

        self.initialize()
        for i_iter in range(niters):
            self.iterate()
            if i_iter % 100 == 0:
                self.dumpit('LDA_word_counts_per_topic.json')
            new_lik = self.loglikelihood()
            print "new likelihood:%s"%new_lik
            if new_lik < old_lik and i_iter > burnin:
                print "converged", "iter #:", i_iter
                return
            old_lik=new_lik


    def printit(self):

        #We first call the base printit methods (prints most frequent terms per topic)
        super(LDAModel,self).printit()

        #Then we print LDA specific (documents per topic)
        ndocs_topics = [0]*self.ntopics            
        for doc_id, doc in enumerate(self.docs):
            for topic in range(self.ntopics):
                ndocs_topics[topic] += self.ntopic_by_doc_topic[doc_id][topic]
        print "Terms per topics :",  "(total)", sum(ndocs_topics), ndocs_topics


    def dumpit(self,fname):
        dumpout = dict([(topic, self.nterm_by_topic_term[topic]) for topic in range(self.ntopics)])
        simplejson.dump(dumpout, open(fname,'w'))


class ARTModel(GraphModel):
    def __init__(self,*args):
        super(ARTModel,self).__init__(*args)
        #prior among topic in pair of (author, recipient)
        self.falpha = 0.5
        #prior among words in topics
        self.fbeta = 0.5


        #initializing the count variables
        #number of times that a topic was assigned to a certain recipient
        self.ntopic_by_recipient_topic = defaultdict(lambda: defaultdict(int))
        #aggregate of the previous attribute per recipient (makes computation faster)
        self.ntopic_by_recipient = defaultdict(int)


        #number of times that a vocabulary word was assigned to a topic
        self.nterm_by_topic_term = [defaultdict(int) for topic in xrange(self.ntopics)]
        #aggregate of the previous attribute per topic (makes computation faster)
        self.nterm_by_topic      = zeros((self.ntopics, ))

        #internal representation of the hidden assignement of topic for all words
        self.z_topic          = [zeros((doc.n_words, ),dtype=int) for doc in self.docs]
        #internal representation of the hidden assignement of recipient for all words
        self.z_recipient          = [zeros((doc.n_words, ),dtype=int) for doc in self.docs]
 
        
    def add_rec(self,recipient_id, topic, qtty=1.):
        self.ntopic_by_recipient_topic[recipient_id][topic] += qtty
        self.ntopic_by_recipient[recipient_id]   += qtty             

    def add_topic(self, topic, term_id, qtty=1.):
        self.nterm_by_topic_term[topic][term_id] += qtty
        self.nterm_by_topic[topic] += qtty
        
    def remove_rec(self, recipient_id, topic):
        self.add_rec(recipient_id, topic,  -1.)

    def remove_topic(self, topic, term_id):
        self.add_topic(topic, term_id,  -1.)


    def loglikelihood(self):
        """this function is most likely broken. do not trust this"""
        loglike = 0.
        for k in xrange(self.ntopics):
            loglike += logdelta(map(operator.add, self.nterm_by_topic_term[k].values(), self.beta.values()))
        loglike -= logdelta(self.beta.values()) * self.ntopics

        for rec in self.ntopic_by_recipient_topic:
            loglike += logdelta(map(operator.add, [ self.ntopic_by_recipient_topic[rec][topic] for topic in range(self.ntopics)], self.alpha))

        num_recipients=len(self.ntopic_by_recipient_topic)
        #TODO: not sure, should we really multitply the logdelta by the number of recipients?
        loglike -= logdelta(self.alpha) * num_recipients
        return loglike
        
        
    def initialize(self):
        """initialize the hidden assignments"""
        self.alpha = [self.falpha / self.ntopics] * self.ntopics
        self.beta  = defaultdict(lambda: self.fbeta / len(self.docs.vocabulary))

        # we randomly assign a topic AND a recipient to every word in every document
        for doc_id, doc in enumerate(self.docs):
            i_word = 0
            for term_id in doc.word_counts:
                for i_term_occ in xrange(doc.word_counts[term_id]):
                    i_topic =  random.choice(range(self.ntopics))
                    i_recipient =  random.choice(doc.recipients)
                    self.z_topic[doc_id][i_word] = i_topic
                    self.z_recipient[doc_id][i_word] = i_recipient
                    #TODO: why does the update equation does not take into account the author?
                    #neither do the count variables...? 
                    self.add_rec(i_recipient, i_topic)
                    self.add_topic(i_topic, term_id)
                    i_word += 1

        

    def iterate(self):
        for doc_id, doc in enumerate(self.docs):
            i_word=0
            for term_id in doc.word_counts:
                for i_term_occ in xrange(doc.word_counts[term_id]):
                    #We first remove our past hidden topic from the counts
                    self.remove_topic(self.z_topic[doc_id][i_word], term_id)
                    self.remove_rec(self.z_recipient[doc_id][i_word], self.z_topic[doc_id][i_word])

                    old_topic = self.z_topic[doc_id][i_word]

                    #Then we find the new params for the new recipients given the old hidden topic
                    param_xi = [(self.ntopic_by_recipient_topic[recipient][old_topic] + self.alpha[old_topic]) / ( self.ntopic_by_recipient[recipient] + self.falpha) for recipient in doc.recipients]
                    # We sample a new recipient
                    new_recipient = doc.recipients[indice(multinomial(1, param_xi))]

                    # We use that new recipient to compute parameters for the new hidden topic
                    param_zi = [(self.nterm_by_topic_term[topic][term_id] + self.beta[term_id]) / ( self.nterm_by_topic[topic] + self.fbeta) * (self.ntopic_by_recipient_topic[new_recipient][topic] +  self.alpha[topic] ) / ( self.ntopic_by_recipient[new_recipient] + self.falpha) for topic in range(self.ntopics)]

                    new_topic = indice(multinomial(1, param_zi))
                    
                    self.add_topic(new_topic, term_id)
                    self.add_rec(new_recipient, new_topic)
                    self.z_topic[doc_id][i_word] = new_topic
                    self.z_recipient[doc_id][i_word] = new_recipient
                    i_word += 1

                    
            if n_verbose_iterate > -1 and ismultiple(doc_id, n_verbose_iterate):
                print " doc : ", doc_id ,"/" , len(self.docs)


    def run(self,niters,burnin = 300):
        old_lik = -999999999999

        self.initialize()
        for i_iter in range(niters):
            print "iteration %s of %s" % (i_iter, niters)
            self.iterate()
            self.dumpit('ART_word_counts_per_topic.json', 'ART_topic_counts_per_recipient.json')
            new_lik = self.loglikelihood()
            print "new likelihood:%s"%new_lik
            if new_lik < old_lik and i_iter > burnin:
                print "converged", "iter #:", i_iter
                return
            old_lik=new_lik


    def printit(self):
        super(ARTModel, self).printit()


    def dumpit(self,fname_topic,fname_recipient):
        dumpout = dict([(topic, self.nterm_by_topic_term[topic]) for topic in range(self.ntopics)])
        simplejson.dump(dumpout, open(fname_topic,'w'))
        simplejson.dump(self.ntopic_by_recipient_topic, open(fname_recipient,'w'))

        
class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
  
if __name__ == "__main__":

    try:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "htw", ["help"])
        except getopt.error, msg:
            raise Usage(msg)
        
        for o, a in opts:
            if o in ("-h", "--help"):
                raise Usage(__file__ +' -alpha ')
            if o in ("-t", "--test"):
                print >>sys.stdout, 'TEST MODE'
                test = True
            if o in ("-w", "--write test"):
                print >>sys.stdout, 'TEST MODE'
                test = True
        if len(args)<0 :
            raise Usage("arguments missing")

        #The creation of this ARTModel assumes the existence of SparsePost
        #model (mongo models)
        model = ARTModel(-1,50,SparsePost)
        model.info()
        model.run(300,300)
        model.printit()
        
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
