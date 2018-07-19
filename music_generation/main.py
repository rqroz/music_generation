from training.data_extraction import *
from training.data_preparation import prepare_sequences
from training.data_modelling import create_network_model

from music21 import stream, converter

import numpy

def generate_encoded_notes(total_notes, model, network_input, pitch_names, n_vocab):
    """
        Generates a sequence of 'total_notes' number of notes based on the given
        network model and input, using the pitch names and the vocabulary size
        to shape the data.

        The list generated is different at each function call as its starting
        point is dictated by a random function.
    """

    # Reverse dictionary mapping number into corresponding notes
    int_to_note = dict((number, note) for number, note in enumerate(pitch_names))

    # Random start note index
    start = numpy.random.randint(0, len(network_input) - 1)

    pattern = network_input[start]

    prediction_output = []

    for note_index in range(total_notes):
        print("Generating note %d"%(note_index+1))
        # Reshape the pattern into a valid network input and normalize it
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        # Get the corresponding prediction
        prediction = model.predict(prediction_input, verbose=0)

        # Figure note index with highest probability, resolve it from int_to_note
        # dictionary and append it to the output.
        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        # Append the current index to the pattern and remove the first element
        # (as if we were right shifting the prediciton_input by 1)
        pattern = numpy.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def decode_note(pattern):
    """
        Decodes a pattern for a note into a Note object from a Piano.
    """
    new_note = note.Note(pattern)
    new_note.storedInstrument = instrument.Piano()
    return new_note

def decode_chord(pattern, offset):
    """
        Decodes a pattern for a chord into a Chord object with various notes.
    """
    notes = []
    for curr_note in pattern.split('.'):
        new_note = decode_note(int(curr_note))
        notes.append(new_note)

    new_chord = chord.Chord(notes)
    new_chord.offset = offset
    return new_chord

def decode_notes_list(encoded_notes):
    """
        Returns a list of Note objects based on the content of encoded_notes list.
        - Decoding:
            Chord Pattern: split the string into an array of string representing
            the notes. Then loop through the string representation of each note
            and create a Note object for them, appending it to the final Chord
            object.
            Note Pattern: create a Note object using the string representation
            of the pitch contained in the pattern.
    """
    offset = 0
    output_notes = []

    for pattern in encoded_notes:
        if ('.' in pattern) or pattern.isdigit(): # if the pattern is a chord
            new_element = decode_chord(pattern, offset)
        else: # pattern is a note
            new_element = decode_note(pattern)

        output_notes.append(new_element)
        offset += 0.5

    return output_notes

def generate_song(total_notes):
    NOTES_FILE_PATH = 'training/data/notes.txt'
    # notes: Set of encoded values related to distinct notes/chords in the dataset
    notes = read_notes_file(NOTES_FILE_PATH) if notes_file_exists(NOTES_FILE_PATH) else generate_notes_set(NOTES_FILE_PATH)

    # Get size of dataset
    n_vocab = len(notes)

    # Get all pitch names
    pitch_names = sorted(set(notes))

    # Resolve network input and output objects
    network_input, network_output = prepare_sequences(notes, n_vocab)

    # Recover network model with trained weights
    model = create_network_model(network_input, n_vocab)
    model.load_weights('weights.hdf5')

    # Generate encoded notes
    encoded_notes = generate_encoded_notes(total_notes, model, network_input, pitch_names, n_vocab)
    print("Encoded Notes:\n%s"%encoded_notes)

    # Generate Chord/Note objects list
    generated_notes = decode_notes_list(encoded_notes)

    midi_stream = stream.Stream(generated_notes)
    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    generate_song(500)
