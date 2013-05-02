# -*- coding : UTF8 -*-
'''
Created on 2013-1-3

@author: zhou yang
'''

import random
import pprint
from util import zeros

class GibbsLDA:
	'''Gibbs sampler for estimating the best assignments of topics for
	words and documents in a corpus. The algorithm is introduced in
	Tom Griffiths' paper "Gibbs sampling in the generative model of 
	Latent Dirichlet Allocation" (2002).
	'''
	def __init__(self, topic_num=2, alpha=5, beta=0.5, ITERATIONS=10000, BURN_IN=2000, 
			THIN_INTERVAL=100, SAMPLE_LAG=10):
		'''Config the LDA model and Gibbs sampler.
		'''
		## Config the parameters of lda
		# topic number
		self.K = topic_num
		# dirichlet parameter of doc-topic
		self.alpha = alpha
		# dirichlet parameter of topic-word
		self.beta = beta
		
		## Config the arguments gibbs sampler usess
		# max iterations
		self.ITERATIONS = ITERATIONS
		# burn in period
		self.BURN_IN = BURN_IN
		# ? what is the difference between thinning interval and sampe lag
		# thinning interval
		self.THIN_INTERVAL = THIN_INTERVAL
		# sample lag
		self.SAMPLE_LAG = SAMPLE_LAG


	def fit(self, documents, V=None):
		'''Fit the data
		'''
		# document data (term lists)
		self.documents = documents
		# vocabulary size
		if V is None:
			self.V = max(max(documents)) + 1
		# document number
		self.M = len(documents)
		
		self.__estimate()
		

	def predict(self, documents, V=None):
		'''Predict the data
		'''
		pass


	def __init_state(self):
		'''Initialisation: Random assignments with equal probabilities
		'''
		## initialise count variables.
		# number of word i assigned to topic j
		self.nw = zeros(self.V, self.K)
		# number of words in document i assigned to topic j.
		self.nd = zeros(self.M, self.K)
		# total number of words assigned to topic j.
		self.nwsum = [0] * self.K
		# total number of words in document i
		self.ndsum = [0] * self.M

		## The z_i are are initialised to values in [1,K] to determine 
		## the initial state of the Markov chain.

		# topic assignments for each word.
		self.z = []
		for m in range(self.M):
			N = len(self.documents[m])
			self.z.append([0] * N)
			for n in range(N):
				topic = int(random.random() * self.K)
				self.z[m][n] = topic
				self.nw[documents[m][n]][topic] += 1
				self.nd[m][topic] += 1
				self.nwsum[topic] += 1
			self.ndsum[m] = N

		## 
		if self.SAMPLE_LAG > 0:
			# cumulative statistics of theta
			self.thetasum = zeros(self.M, self.K)
			self.theta = zeros(self.M, self.K)
			# cumulative statistics of phi
			self.phisum = zeros(self.K, self.V)
			self.phi = zeros(self.K, self.V)
			# size of statistics
			self.numstats = 0

		self.__print_init_state()


	def __print_init_state(self):
		print '##init state: '
		print 'word i assigned to topic j'
		pprint.pprint(self.z)
		print 'word i assigned to topic j count'
		pprint.pprint(self.nw)
		print 'word in document i assigned to topic j count'
		pprint.pprint(self.nd)
		print 'all words assigned to topic j count'
		pprint.pprint(self.nwsum)
		print 'length of document j'
		pprint.pprint(self.ndsum)


	def __update_params(self):
		'''Update the parameters of LDA model
		'''
		for m in range(self.M):
			for k in range(self.K):
				#import pdb; pdb.set_trace()
				self.thetasum[m][k] += (self.nd[m][k]+self.alpha) / \
							(self.ndsum[m] + self.K*self.alpha)
		#print 'in update: self.thetasum'
		#print self.thetasum

		for k in range(self.K):
			for w in range(self.V):
				self.phisum[k][w] += (self.nw[w][k]+self.beta) / \
							(self.nwsum[k] + self.V*self.beta)

		self.numstats += 1

	def __sampling(self, m, n):
		'''Samping from full probability distribution
		'''
		# remove z_i from the count variables
		#import pdb; pdb.set_trace()
		topic = self.z[m][n]
		self.nw[self.documents[m][n]][topic] -= 1
		self.nd[m][topic] -= 1
		self.nwsum[topic] -= 1
		self.ndsum[m] -= 1

		# do multinomial sampling via cumulative method
		p = [0] * self.K
		for k in range(self.K):
			p[k] = (self.nw[self.documents[m][n]][k] + self.beta) \
						/ (self.nwsum[k] + self.V*self.beta)   \
				* (self.nd[m][k]+self.alpha)  \
					/ (self.ndsum[m] + self.K*self.alpha)
		# cumulate multinomial parameters
		for k in range(1, len(p)):
			p[k] += p[k-1]
		# scaled sample because of unnormalised p[]
		u = random.random() * p[self.K-1]
		for topic in range(len(p)):
			if u < p[topic]:
				break

		# add newly estimated z_i to count variables
		self.nw[self.documents[m][n]][topic] += 1
		self.nd[m][topic] += 1
		self.nwsum[topic] += 1
		self.ndsum[m] += 1

		return topic


	def __estimate(self):
		'''Start estimating gibbs sampling 
		'''
		print "Sampling %d iterations with burn-in of %d (B/S=%d)" \
				% (self.ITERATIONS, self.BURN_IN, self.THIN_INTERVAL)

		self.__init_state()

		for i in range(self.ITERATIONS):
			# one scan of all z_i
			for m in range(len(self.z)):
				for n in range(len(self.z[m])):
					# (z_i = z[m][n])
					# sample from p(z_i|z_-i, w)
					topic = self.__sampling(m, n)
					self.z[m][n] = topic

			# get statistics after burn-in
			if i > self.BURN_IN and i%self.SAMPLE_LAG == 0:
				self.__update_params()

		# nornalize theta and phi
		assert self.SAMPLE_LAG  > 0
		if self.SAMPLE_LAG > 0:
			#import pdb; pdb.set_trace()
			self.__compute_theta()
			self.__compute_phi()

		print 'Estimated paramete value is:'
		pprint.pprint(self.theta)
		pprint.pprint(self.phi)

	def __compute_theta(self):
		'''
		'''
		for m in range(self.M):
				for k in range(self.K):
					self.theta[m][k] = self.thetasum[m][k] / self.numstats

	def __compute_phi(self):
		for k in range(self.K):
				for w in range(self.V):
					self.phi[k][w] = self.phisum[k][w] / self.numstats

	def usage(self):
		'''Print the usage information
		'''
		pass

if __name__ == '__main__':
	# words in documents
	documents = [ 
			[1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 6],
			[2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2],
			[1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 0],
			[5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0],
			[2, 2, 4, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 0],
			[5, 4, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2],]

	# topics num.
	K = 2

	alpha = 2
	beta = 0.5

	print 'Latent Dirichlet Allocation using Gibbs Sampling.'

	lda = GibbsLDA(topic_num=K, alpha=alpha, beta=beta, ITERATIONS=100000,\
			 BURN_IN=2000, THIN_INTERVAL=100, SAMPLE_LAG=10)
	lda.fit(documents)

