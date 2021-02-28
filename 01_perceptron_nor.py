#!/usr/bin/python3
import numpy as np
from itertools import repeat, product
from operator import xor, not_
from functools import reduce
import random

class Perceptron:

    # idim -> input dimension
    def __init__(self, idim, randomize=True, threshold=0, bias=1, eta=1, positive_output=1, negative_output=-1):
        if randomize:
            # bias weight
            self.weights = np.random.rand(idim + 1)
        else:
            # bias weight
            self.weights = np.zeros(idim + 1)
        self.threshold = threshold
        self.po = positive_output
        self.no = negative_output
        self.eta = eta
        self.bias = bias

    def infer(self, ip):
        # create dot product
        # add the bias input
        op = self.weights.dot((self.bias, *ip))
        # if the result is above the threshold, the output
        # is positive
        if op > self.threshold:
            return self.po
        else:
            return self.no

    # input_x: inputs on which the output of the perceptron
    # should be x
    def train(self, input_negative, input_positive):
        # append the corresponding output values to
        # each input
        y_g = []
        # prepend the bias input to each of the training
        # inputs
        for i_n in input_negative:
            y_g.append(((self.bias, *i_n), self.no))
        for i_p in input_positive:
            y_g.append(((self.bias, *i_p), self.po))
        # the set of correctly guessed inputs
        correct = set()
        # condition for convergence
        # unless we get correct result for every training
        # input, continue training
        while len(correct) < len(y_g):
            # pick a random input
            y_t = np.random.randint(0, len(y_g))
            y_t = y_g[y_t]
            # get the output
            s = self.weights.dot(y_t[0])
            if s > self.threshold:
                o = self.po
            else:
                o = self.no
            # if the inferred output is not equal to
            # our expected output, we adjust
            if o != y_t[1]:
                self.weights += (self.eta * (y_t[1] - o) * np.array(y_t[0]))
                # reset the whole set
                # so that we have to retry for
                # everyone else
                correct.clear()
            else:
                # otherwise, we add this to the set
                # of correctly guessed outputs
                correct.add(y_t)

def and_(*args):
    for a in args:
        if not a:
            return 0
    return 1

def or_(*args):
    for a in args:
        if a:
            return 1
    return 0

def xor(*args):
    c = 0
    for a in args:
        if a:
            c += 1
    return c & 1

def nand(*args):
    return not_(and_(*args))

def nor(*args):
    return not_(or_(*args))

def xnor(*args):
    return not_(xor(*args))

def gen_training_input(dim, func):
    nums = list(product([0, 1], repeat=dim))
    positive_inputs = []
    negative_inputs = []
    for i in nums:
        if func(*i):
            positive_inputs.append(i)
        else:
            negative_inputs.append(i)
    return positive_inputs, negative_inputs

def main():
    dim = 3
    func = nor
    p = Perceptron(dim, False)
    print("Generating inputs..")
    positive_input, negative_input = gen_training_input(dim, func)
    print("Before training, weights:", p.weights)
    p.train(negative_input, positive_input)
    print("After training, weights:", p.weights)
    p_test = random.choice(positive_input)
    n_test = random.choice(negative_input)
    p_res = p.infer(p_test)
    print("Result on", p_test, ":", p_res, end='\t')
    if p_res:
        print("Correct Output")
    else:
        print("Incorrect Output!")
    n_res = p.infer(n_test)
    print("Result on", n_test, ":", n_res, end='\t')
    if n_res == -1:
        print("Correct Output")
    else:
        print("Incorrect Output!")

if __name__ == "__main__":
    main()
