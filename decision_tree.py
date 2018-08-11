import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None 

		self.feature_uniq_split = None 


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			sum_total = np.sum(branches)
			res = 0
			ent_brs, sum_brs = [0] * len(branches[0]), [0] * len(branches[0])
			for idx_br in range(len(branches[0])):
				sum_br = np.sum(branches[:,idx_br])
				sum_brs[idx_br] = sum_br
				sum_ = 0
				for item in branches[:,idx_br]:
					sum_ += (item/sum_br)*np.log((item/sum_br)) if item != 0 else 0
				ent_brs[idx_br] = (-1)*sum_
			sum_brs = np.divide(sum_brs, sum_total)

			res = np.dot(ent_brs, sum_brs)
			return res

		minn = float('inf')
		min_idx = 0
		for idx_dim in range(len(self.features[0])):
			num_br = int(np.max(np.array(self.features)))+1
			feature_ = np.zeros((self.num_cls, num_br))
			for y_,br_ in zip(self.labels, np.array(self.features)[:,idx_dim]):
				feature_[y_][int(br_)] += 1

			cur = conditional_entropy(feature_)
			if cur < minn:
				minn = cur
				min_idx = idx_dim
				min_feat = feature_

		best_feat = np.array(self.features)[:,min_idx]
		self.dim_split = min_idx 
		self.feature_uniq_split = np.unique(best_feat).tolist()

		best_feat = np.array(self.features)[:,min_idx]
		feat_dict = {cla: (best_feat==cla).nonzero()[0] for cla in np.unique(best_feat)}

		for val in feat_dict.values():
			y_subset = np.array(self.labels).take(val, axis=0).tolist()
			x_subset = np.array(self.features).take(val, axis=0).tolist()
			num_cl = np.max(y_subset)+1
			child = TreeNode(x_subset, y_subset, num_cl)

			prev = x_subset[0]
			flag = True
			for item in x_subset:
				if item != prev:
					flag = False
			if flag:
				child.splittable = False

			self.children.append(child)

		for child in self.children:
			if child.splittable:
				child.split()
		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



