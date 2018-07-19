import numpy
from keras.utils import np_utils

def prepare_sequences(notes, n_vocab):
    """
        SEQUENCE_LENGTH: Defines the 'memory size' of the LSTM Network. In other
                         words, this will define how many previous notes the LSTM
                         Network will have to help in the next note prediciton.

    """
    SEQUENCE_LENGTH = 100

    # Get all pitch names
    pitch_names = sorted(set(notes))

    """
        note_to_int: Dictionary mapping pitches to integers.
                     The network performs much better with integer-based numerical
                     data than string-based categorial data.
    """
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    network_input = []
    network_output = []

    # Create input sequences and the corresponding outputs
    for i in range(len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i : i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[ch] for ch in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    # Normalizing input
    network_input = network_input / float(n_vocab)

    # Transforming output to categorical binary matrix (one-hot encoding)
    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output
