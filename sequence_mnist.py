"""
Provides the sequenced MNIST handwritten digit dataset.
"""
# standard libraries
from __future__ import print_function
import math
# third party libraries
import numpy
# internal imports
from opendeep.data import MNIST


def get_sequenced_mnist(sequence_number=0, seq_3d=False, seq_length=30, rng=None, flatten=True):
    """
        sequence_number : int, optional
            The sequence method to use if we want to put the input images into a specific order. 0 defaults to random.
        seq_3d : bool, optional
            When sequencing, whether the output should be
            3D tensors (batches, subsequences, data) or 2D (sequence, data).
        seq_length: int, optional
            The length of subsequences to split the data.
        rng : random, optional
            The random number generator to use when sequencing.
    """
    mnist = MNIST(flatten=flatten)

    # sequence the dataset
    if sequence_number is not None:
        _sequence(mnist, sequence_number=sequence_number, rng=rng)

    # optionally make 3D instead of 2D
    if seq_3d:
        print("Making 3D....")
        # chop up into sequences of length seq_length
        # first make sure to chop off the remainder of the data so seq_length can divide evenly.
        if mnist.train_inputs.shape[0] % seq_length != 0:
            length, dim = mnist.train_inputs.shape
            if mnist.train_targets.ndim == 1:
                ydim = 1
            else:
                ydim = mnist.train_targets.shape[-1]
            mnist.train_inputs = mnist.train_inputs[:seq_length * int(math.floor(length / seq_length))]
            mnist.train_targets = mnist.train_targets[:seq_length * int(math.floor(length / seq_length))]
            # now create the 3D tensor of sequences - they will be (num_sequences, sequence_size, 784)
            mnist.train_inputs = numpy.reshape(mnist.train_inputs, (length / seq_length, seq_length, dim))
            mnist.train_targets = numpy.reshape(mnist.train_targets, (length / seq_length, seq_length, ydim))

        if mnist.valid_inputs.shape[0] % seq_length != 0:
            length, dim = mnist.valid_inputs.shape
            if mnist.valid_targets.ndim == 1:
                ydim = 1
            else:
                ydim = mnist.valid_targets.shape[-1]
            mnist.valid_inputs = mnist.valid_inputs[:seq_length * int(math.floor(length / seq_length))]
            mnist.valid_targets = mnist.valid_targets[:seq_length * int(math.floor(length / seq_length))]
            # now create the 3D tensor of sequences - they will be (num_sequences, sequence_size, 784)
            mnist.valid_inputs = numpy.reshape(mnist.valid_inputs, (length / seq_length, seq_length, dim))
            mnist.valid_targets = numpy.reshape(mnist.valid_targets, (length / seq_length, seq_length, ydim))

        if mnist.test_inputs.shape[0] % seq_length != 0:
            length, dim = mnist.test_inputs.shape
            if mnist.test_targets.ndim == 1:
                ydim = 1
            else:
                ydim = mnist.test_targets.shape[-1]
            mnist.test_inputs = mnist.test_inputs[:seq_length * int(math.floor(length / seq_length))]
            mnist.test_targets = mnist.test_targets[:seq_length * int(math.floor(length / seq_length))]
            # now create the 3D tensor of sequences - they will be (num_sequences, sequence_size, 784)
            mnist.test_inputs = numpy.reshape(mnist.test_inputs, (length / seq_length, seq_length, dim))
            mnist.test_targets = numpy.reshape(mnist.test_targets, (length / seq_length, seq_length, ydim))

    print('Train shape is: {!s}, {!s}'.format(mnist.train_inputs.shape, mnist.train_targets.shape))
    print('Valid shape is: {!s}, {!s}'.format(mnist.valid_inputs.shape, mnist.valid_targets.shape))
    print('Test shape is: {!s}, {!s}'.format(mnist.test_inputs.shape, mnist.test_targets.shape))
    return mnist

def _sequence(mnist, sequence_number, rng=None):
    """
    Sequences the train, valid, and test datasets according to the artificial sequences I made up...

    Parameters
    ----------
    sequence_number : {0, 1, 2, 3, 4}
        The sequence is is determined as follows:

        ======  =================================================
        value   Description
        ======  =================================================
        0       The original image ordering.
        1       Order by digits 0-9 repeating.
        2       Order by digits 0-9-9-0 repeating.
        3       Rotates digits 1, 4, and 8. See implementation.
        4       Has 3 bits of parity. See implementation.
        ======  =================================================

    rng : random
        the random number generator to use
    """
    print("Sequencing MNIST with sequence {!s}".format(sequence_number))
    if rng is None:
        rng = numpy.random
        rng.seed(1)

    # Find the order of MNIST data going from 0-9 repeating if the first dataset
    train_ordered_indices = None
    valid_ordered_indices = None
    test_ordered_indices  = None
    if sequence_number == 0:
        pass
    elif sequence_number == 1:
        train_ordered_indices = _sequence1_indices(mnist.train_targets)
        valid_ordered_indices = _sequence1_indices(mnist.valid_targets)
        test_ordered_indices  = _sequence1_indices(mnist.test_targets)
    elif sequence_number == 2:
        train_ordered_indices = _sequence2_indices(mnist.train_targets)
        valid_ordered_indices = _sequence2_indices(mnist.valid_targets)
        test_ordered_indices  = _sequence2_indices(mnist.test_targets)
    elif sequence_number == 3:
        train_ordered_indices = _sequence3_indices(mnist.train_targets)
        valid_ordered_indices = _sequence3_indices(mnist.valid_targets)
        test_ordered_indices  = _sequence3_indices(mnist.test_targets)
    elif sequence_number == 4:
        train_ordered_indices = _sequence4_indices(mnist.train_targets)
        valid_ordered_indices = _sequence4_indices(mnist.valid_targets)
        test_ordered_indices  = _sequence4_indices(mnist.test_targets)
    else:
        print("MNIST sequence number {!s} not recognized, leaving dataset as-is.".format(sequence_number))

    # Put the data sets in order
    if train_ordered_indices is not None and valid_ordered_indices is not None and test_ordered_indices is not None:
        mnist.train_inputs = mnist.train_inputs[train_ordered_indices]
        mnist.train_targets = mnist.train_targets[train_ordered_indices]
        mnist.valid_inputs = mnist.valid_inputs[valid_ordered_indices]
        mnist.valid_targets = mnist.valid_targets[valid_ordered_indices]
        mnist.test_inputs  = mnist.test_inputs[test_ordered_indices]
        mnist.test_targets  = mnist.test_targets[test_ordered_indices]

    # re-set the sizes
    _train_shape = mnist.train_inputs.shape
    _valid_shape = mnist.valid_inputs.shape
    _test_shape = mnist.test_inputs.shape
    print('Sequence train shape is: {!s}'.format(_train_shape))
    print('Sequence valid shape is: {!s}'.format(_valid_shape))
    print('Sequence test shape is: {!s}'.format(_test_shape))

def _sequence1_indices(labels, classes=10):
    # make sure labels are integers
    labels = [label.astype('int32') for label in labels]
    # Creates an ordering of indices for this MNIST label series (normally expressed as y in dataset)
    # that makes the numbers go in order 0-9....
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    # organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    # draw from each pool (also with the random number insertions) until one is empty
    stop = False
    # check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print("stopped early from dataset1 sequencing - missing some class of labels")
    while not stop:
        # for i in range(classes)+range(classes-2,0,-1):
        for i in range(classes):
            if not stop:
                if len(pool[i]) == 0:  # stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[i].pop())
    return sequence

# order sequentially up then down 0-9-9-0....
def _sequence2_indices(labels, classes=10):
    # make sure labels are integers
    labels = [label.astype('int32') for label in labels]
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    # organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    # draw from each pool (also with the random number insertions) until one is empty
    stop = False
    # check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print("stopped early from dataset2a sequencing - missing some class of labels")
    while not stop:
        for i in range(classes)+range(classes-1,-1,-1):
            if not stop:
                if len(pool[i]) == 0:  # stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[i].pop())
    return sequence

def _sequence3_indices(labels, classes=10):
    # make sure labels are integers
    labels = [label.astype('int32') for label in labels]
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    # organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    # draw from each pool (also with the random number insertions) until one is empty
    stop = False
    # check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print("stopped early from dataset3 sequencing - missing some class of labels")
    a = False
    while not stop:
        for i in range(classes):
            if not stop:
                n=i
                if i == 1 and a:
                    n = 4
                elif i == 4 and a:
                    n = 8
                elif i == 8 and a:
                    n = 1
                if len(pool[n]) == 0:  # stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[n].pop())
        a = not a

    return sequence

# extra bits of parity
def _sequence4_indices(labels, classes=10):
    # make sure labels are integers
    labels = [label.astype('int32') for label in labels]

    def even(n):
        return n % 2 == 0

    def odd(n):
        return not even(n)

    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    # organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    # draw from each pool (also with the random number insertions) until one is empty
    stop = False
    # check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            print("stopped early from dataset4 sequencing - missing some class of labels")
    s = [0, 1, 2]
    sequence.append(pool[0].pop())
    sequence.append(pool[1].pop())
    sequence.append(pool[2].pop())
    while not stop:
        if odd(s[-3]):
            first_bit = (s[-2] - s[-3]) % classes
        else:
            first_bit = (s[-2] + s[-3]) % classes
        if odd(first_bit):
            second_bit = (s[-1] - first_bit) % classes
        else:
            second_bit = (s[-1] + first_bit) % classes
        if odd(second_bit):
            next_num = (s[-1] - second_bit) % classes
        else:
            next_num = (s[-1] + second_bit + 1) % classes

        if len(pool[next_num]) == 0:  # stop the procedure if you are trying to pop from an empty list
            stop = True
        else:
            s.append(next_num)
            sequence.append(pool[next_num].pop())

    return sequence
