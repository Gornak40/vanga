#!/usr/bin/python3
import numpy as np
from pprint import pprint
from random import randrange


class Vanga:
	def __init__(self, step=1):
		self.step = step

	def go(self, x):
		'''calcs powers of x for O(n)'''
		res = np.ones(self.n)
		p = 0
		for i in range(self.n):
			res[i] = pow(x, p)
			p += self.step
		return res

	def train(self, data):
		'''trains the model with (x, y) data for O(n ** 3)'''
		if isinstance(data, list):
			data = np.array(data)
		self.X = data[:,0]
		self.Y = data[:,1]
		self.n = len(self.X)
		self.M = np.zeros((self.n, self.n))
		for i, x in enumerate(self.X):
			self.M[i] = self.go(x)
		self.W = np.linalg.solve(self.M, self.Y)

	def predict(self, x):
		'''makes prediction f(x) for O(n)'''
		return (self.go(x) * self.W).sum()


class NoiseVanga(Vanga):
	def __init__(self, n, step=1):
		self.n = n
		self.step = step

	def take_random(self, arr):
		'''takes random element from array for O(1)'''
		ind = randrange(len(arr))
		arr[ind], arr[-1] = arr[-1], arr[ind]
		return arr.pop()

	def train(self, data, epoch):
		'''trains the model with large (x, y) data for O(epoch * n ** 3)'''
		self.W = np.zeros(self.n)
		for ep in range(epoch):
			arr = [self.take_random(data) for _ in range(self.n)]
			V = Vanga(self.step)
			V.train(arr)
			self.W += V.W
			data += arr
		self.W /= epoch
		pprint(self.W)