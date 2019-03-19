from music21.midi import MidiFile
from music21 import *

if __name__ == '__main__':
    midi = MidiFile()
    print("Hello")
    result = midi.open("k49.mid")
    midi.read()

    track1 = midi.tracks[1]

    timePassed = 0

    notes = []
    currentNotes = []
    currentMeasure = 0
    for event in track1.events:
        if event.time is not None:
            timePassed += event.time
        elif event.type == "NOTE_ON":
            currentNotes.append(event.parameter1)
        if timePassed > midi.ticksPerQuarterNote * 4:
            timePassed -= midi.ticksPerQuarterNote * 4
            notes.append(currentNotes)
            currentNotes = []


    print(notes)

    for index in range(len(notes)):
        c = chord.Chord(notes[index])

        print("Measure %d: %s. Notes(%s)" % (index, c.commonName,c))


    print("end")
    # track0 = midi.MidiTrack(0)
