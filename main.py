#!/usr/bin/python3
from vanga import *
from math import sqrt
from random import random, shuffle, randint


def f(x):
	return x ** 2 - 5 * sqrt(x) + 9


X = list(range(100))
shuffle(X)
data = [(x, f(x)) for x in X[:50]]
NV = NoiseVanga(n=10, step=.5)
NV.train(data, epoch=10)
print(X[-1], NV.predict(X[-1]))
