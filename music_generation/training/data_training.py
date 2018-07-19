from data_extraction import *
from data_preparation import prepare_sequences
from data_modelling import create_network_model

from keras.callbacks import ModelCheckpoint

def train(model, network_input, network_output):
    filepath = 'weights-improvement/{epoch:02d}-{loss:.4f}-bigger.hdf5'
    print("Callback filepath: %s"%filepath)
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')

    callbacks_list = [checkpoint]
    print("callbacks_list: %s"%callbacks_list)

    # Perform Network Traning by Keras module
    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)


def train_network():
    """
        Trains a Neural Network to generate music
    """
    NOTES_FILE_PATH = 'data/notes.txt'
    # notes: Set of encoded values related to distinct notes/chords in the dataset
    notes = read_notes_file(NOTES_FILE_PATH) if notes_file_exists(NOTES_FILE_PATH) else generate_notes_set(NOTES_FILE_PATH)

    # Get size of dataset
    n_vocab = len(notes)
    print("n_vocab: %s"%n_vocab)

    network_input, network_output = prepare_sequences(notes, n_vocab)
    print("Network Input:\n%s"%network_input)
    print("Network Output:\n%s"%network_output)

    model = create_network_model(network_input, n_vocab)
    print("Model: %s"%model)

    train(model, network_input, network_output)
    print("Network Trained!")


if __name__ == '__main__':
    train_network()
