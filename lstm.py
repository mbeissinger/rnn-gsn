import numpy
from theano.tensor import tensor3, matrix
from opendeep.models import Prototype, LSTM, Dense
from sequence_mnist import get_sequenced_mnist

rng = numpy.random
rng.seed(1)

def main():
    seq_len = 30
    dim = 784

    mnist_seq = get_sequenced_mnist(sequence_number=1, seq_3d=True, seq_length=seq_len, flatten=True, rng=rng)

    # 3d input - batch, sequence, dim
    xs = tensor3("sequences")
    ins = ((None, seq_len, dim), xs)

    lstm = LSTM(inputs=ins, hiddens=1024)
    recon = Dense(inputs=lstm, outputs=dim)




if __name__ == '__main__':
    main()
