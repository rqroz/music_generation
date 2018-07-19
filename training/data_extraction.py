from music21 import converter, instrument, note, chord
import os, glob

"""
Notes

 Pitch: Frequency of the sound, represents how low/high the sound.
        Represented by the letters A through G in descending order.

 Octave: Refers to the set of pitches used in a piano.

 Offset: Refers to where the note is located in the piece.

 Chord: Container for a set of notes played at the same time.
 """

NOTES_FILE_PATH = 'data/notes.txt'

def generate_notes_set():
    notes = set()

    for idx, filename in enumerate(glob.glob('midi_songs/*.mid')):
        print("Extracting data %d from %s"%(idx+1, filename))
        midi = converter.parse(filename)
        notes_to_parse = []

        parts = instrument.partitionByInstrument(midi)

        if parts: # If the file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # the file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        """
            Note Encoding: The proccess bellow will encode each element in order to
                           allow us to easily decode the output generated by the
                           network into the corrected notes/chords.
        """
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                """
                    The most significant parts of a note object can be recreated
                    using the string notation of the pitch. Thus, we append this
                    notation to the notes array.
                """
                notes.add(str(element.pitch))
            elif isinstance(element, chord.Chord):
                """
                    Append every chord by encoding the id of every note in the chord
                    together in a string, with notes separated by a dot.
                """
                notes.add('.'.join(str(n) for n in element.normalOrder))

    # Outputing the results to a text file
    with open(NOTES_FILE_PATH, 'w') as notes_file:
        for n in notes:
            notes_file.write("%s\n"%n)

    return list(notes)


def read_notes_file():
    notes = []
    with open(NOTES_FILE_PATH, 'r') as notes_file:
        for line in notes_file:
            encoding = line.replace('\n', '')
            notes.append(encoding)

    return notes

def notes_file_exists():
    return os.path.isfile(NOTES_FILE_PATH)
