import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] 
		self.betas = []       
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		results = np.zeros(len(features),)
		for i in range(self.T):
			h_i = self.clfs_picked[i].predict(features)
			beta_h = self.betas[i]*np.array(h_i)
			results = results + beta_h
		for i, res in enumerate(results):
			results[i] = 1 if res >= 0 else -1
		return list(results.flatten())
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		w = np.divide(np.ones(len(features),), len(features))

		for iter in range(self.T):
			min_ = float('inf')
			min_set = None
			for clf in self.clfs:
				sum_ = 0
				feature_ = clf.predict(features)
				for i, feat in enumerate(feature_):
					if labels[i] != feat:
						sum_ += w[i]*(1)
				if sum_ < min_:
					min_ = sum_
					min_set = clf
			beta_ = np.log((1-min_)/min_) / 2.0
			feature_ = min_set.predict(features)
			for i, feat in enumerate(feature_):
				if labels[i] != feat:
					w[i] = w[i]*np.exp(beta_)
				else:
					w[i] = w[i]*np.exp(-beta_)
			sum_w = np.sum(w)
			w = np.divide(w, sum_w)

			self.clfs_picked.append(min_set)
			self.betas.append(beta_)
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		pi = np.divide(np.ones(len(features)), 2)
		fx = np.zeros(len(features),)

		for iter in range(self.T):
			w = np.multiply(pi, (1-pi))
			z = np.divide((0.5*(np.array(labels)+1) - pi), w)
			min_ = float('inf')
			min_set = None
			for clf in self.clfs:
				sum_ = 0
				feature_ = clf.predict(features)
				sum_ = np.dot(w, np.power(z-feature_,2))
				if sum_ < min_:
					min_ = sum_
					min_set = clf
			feature_ = min_set.predict(features)
			fx = np.add(fx, 0.5*np.array(feature_))
			pi = np.divide(1, (1+np.exp(-2*fx)))

			self.clfs_picked.append(min_set)
			self.betas.append(0.5)
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	