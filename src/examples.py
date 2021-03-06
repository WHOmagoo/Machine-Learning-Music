import random

from music21.midi import MidiFile
from music21.midi import MidiTrack
from music21.midi import MidiEvent
from music21.midi import DeltaTime
from music21.midi import ChannelVoiceMessages
from music21.midi import MetaEvents
from music21 import *
import os
import numpy as np

from timeit import default_timer as timer
time_spent = [0,0,0,0,0,0,0]


num_notes = 65
zeroes_output = [[0] * num_notes]
zeroes_output[0][num_notes - 1] = 1
notes_per_chord = 3
classes = np.zeros((num_notes,num_notes))

for i in range(num_notes):
    classes[i][i] = 1

def get_class(classNum):
    return classes[classNum]



def get_next_events(tracks):
    lowest_index = 0
    lowestTime = -1.1

    for index in range(len(tracks)):
        cur_event = tracks[index].events[0]
        if cur_event.type == 'DeltaTime':
            if cur_event.time == 0:
                # TODO pop items and adjust other tracks
                lowest_index = index
                lowestTime = 0
                break
            elif lowestTime == -1.1 or cur_event.time < lowestTime:
                lowestTime = tracks[index].events[0].time
                lowest_index = index
        else:
            print("No leading DeltaTime in track", index)

    next_events = []
    cur_track_events = tracks[lowest_index].events
    passed_first_delta_time = False

    for index in range(len(cur_track_events)):
        cur_event = cur_track_events[0]
        if cur_event.type == 'DeltaTime':
            if passed_first_delta_time:
                break
            else:
                passed_first_delta_time = True

        if cur_event.type in ['DeltaTime', ChannelVoiceMessages.NOTE_ON, ChannelVoiceMessages.NOTE_OFF] or lowest_index == 0:
            next_events.append(cur_event)

        cur_track_events.remove(cur_event)

        if cur_event.type == midi.MetaEvents.END_OF_TRACK and (lowest_index != 0 or len(tracks) == 0):
            # tracks.remove(tracks[lowest_index])
            # If we removed from the last track and reached END_OF_TRACK break out of the loop and continue
            # as the rest of the code will remove this track and return its contents
            # Otherwise remove time from other tracks
            removeTime(lowestTime, lowest_index, tracks)

            tracks.remove(tracks[lowest_index])

            # If the next events only contains delta time return nothing, otherwise return stored events
            if len(next_events) > 1:
                next_events.remove(cur_event)
                return next_events
            else:
                return []

    contains_non_delta_time_item = False
    for item in next_events:
        if item.type != "DeltaTime":
            contains_non_delta_time_item = True

    removeTime(lowestTime, lowest_index, tracks)

    if len(tracks[lowest_index].events) == 0:
        tracks.remove(tracks[lowest_index])

    if contains_non_delta_time_item:
        return next_events
    else:
        return []


def removeTime(amount_of_time, index_to_skip, tracks):
    # adjust the following delta times by the amount removed
    if amount_of_time > 0:
        for index in range(len(tracks)):
            if index == index_to_skip:
                pass
            else:
                cur_event = tracks[index].events[0]
                if cur_event.type == 'DeltaTime':
                    if cur_event.time > 0:
                        cur_event.time -= amount_of_time

def merge_tracks(tracks):
    # longest = 0
    # for track in tracks:
    #     if len(track.events)  > longest:
    #         longest = len(track.events)

    merged_events = []
    time_passed = 0

    while True:
        result = get_next_events(tracks)

        for item in result:
            merged_events.append(item)
            if item.type == 'DeltaTime':
                time_passed += item.time

        if len(tracks) == 0:
            break

    midi_track = MidiTrack(0)
    midi_track.events = merged_events
    #TODO check effects of this
    #midi_track.length = time_passed

    for item in midi_track.events:
        item.track = midi_track
        if item.channel is not None:
            item.channel = 1

    return [midi_track]


def get_tick_length_of_note(track, index_of_start):
    event = track.events[index_of_start]
    if event.type != ChannelVoiceMessages.NOTE_ON:
        return None

    length_of_note = 0

    for index in range(index_of_start + 1, len(track.events)):
        current_event = track.events[index]
        if current_event.type == 'DeltaTime':
            length_of_note += current_event.time
        elif current_event.pitch == event.pitch:
            if current_event.type == ChannelVoiceMessages.NOTE_OFF:
                break
            elif current_event.type == ChannelVoiceMessages.NOTE_ON and current_event.velocity == 0:
                current_event.type = ChannelVoiceMessages.NOTE_OFF
                break

    return length_of_note


def get_note_type(ticks_per_quarter_note, tick_length, round_nearest=False):
    sixteenth_note_length = ticks_per_quarter_note / 4.0

    length_in_sixteenth_notes = tick_length / sixteenth_note_length

    length_rounded = round(length_in_sixteenth_notes)

    if length_rounded - .5 <= length_in_sixteenth_notes <= length_rounded + .5:
        return length_rounded
    else:

        if round_nearest:
            return length_rounded
        else:
            print("Approximation Error")
            # TODO come up with a good approximation algorithm, either remove entirely or round to nearest 16th note
            return 0


longest = 1024


def array_append(array, value, index):
    if index >= len(array):
        global longest
        longest *= 2
        result = [[]] * longest
        result[:len(array)] = array
        result[index] = value
        return result
    else:
        array[index] = value
        return array


def quantize_midi(track, ticks_per_quarternote):
    # result, each index is one 16th note, each item will be a list of notes that are started at the time along with their length
    result = [-1] * longest
    result[0] = []
    current_time = 0
    current_time_exact = 0

    cur_index = 0
    for note_index in range(len(track.events)):
        if track.events[note_index].type == "DeltaTime":
            current_time_exact += track.events[note_index].time
            new_time_passed = get_note_type(ticks_per_quarternote, current_time_exact, True)
            for i in range(current_time, new_time_passed):
                cur_index += 1
                result = array_append(result, [], cur_index)
                #result.append([])

            current_time = new_time_passed

        length = get_tick_length_of_note(track, note_index)
        if length is not None and length > 0:
            note_type = get_note_type(ticks_per_quarternote, length)
            if note_type is not None and note_type > 0 and track.events[note_index].pitch in range(21+num_notes//2,88+21-num_notes//2):
                tuple = [track.events[note_index].pitch - (21+num_notes//2) + 1, note_type]
                result[current_time].append(tuple)

    resulting_size = len(result)
    while len(result) > 0 and (result[resulting_size-1] == [] or result[resulting_size-1] == -1):
        resulting_size -= 1

    return result[:resulting_size]


def quantized_to_midi(quantized_data):
    resulting_midi = MidiFile()
    resulting_midi.format = 0
    resulting_midi.ticksPerQuarterNote = 400

    track = MidiTrack(0)
    track.events.append(MidiEvent())
    track.setChannel(1)


def write_starting_messages(track):
    no_time_spacing = DeltaTime(track, time=0)

    track.events.append(no_time_spacing)

    channel_prefix = MidiEvent(track=track, type=midi.MetaEvents.MIDI_CHANNEL_PREFIX)
    channel_prefix.data = b'\x00'
    track.events.append(channel_prefix)

    track.events.append(no_time_spacing)

    track_name = MidiEvent(track=track, type=MetaEvents.SEQUENCE_TRACK_NAME)
    track_name.data = bytes("Quantized Midi", 'ascii')
    track.events.append(track_name)

    # track.events.append(no_time_spacing)

    # smtpe_offset = MidiEvent(track=track, type='SMTPE_OFFSET')
    # smtpe_offset.data = b'\x00\x00\x00\x00'
    # track.events.append(smtpe_offset)
    #
    # track.events.append(no_time_spacing)

    # SET_TEMPO should be MetaEvents.SET_TEMPO
    # tempo = MidiEvent(track=track, type='SET_TEMPO')
    # tempo.data = b'\x05\xe8\x19'#b'\x07\xA1\x20'
    # track.events.append(tempo)

    # track.events.append(no_time_spacing)
    #
    # time_signature = MidiEvent(track=track, type='TIME_SIGNATURE')
    # time_signature.data = b'\x04\x02\x18\x08'
    # track.events.append(time_signature)

    track.events.append(no_time_spacing)

    program_change = MidiEvent(track=track, type=ChannelVoiceMessages.PROGRAM_CHANGE)
    program_change.channel = 1
    program_change.data = 1
    track.events.append(program_change)

    # track.events.append(no_time_spacing)
    #
    # Controller change should be ChannelVoiceMessages.CONTROLLER_CHANGE now
    # controller_change = MidiEvent(track=track, type='CONTROLLER_CHANGE')
    # controller_change.channel = 1
    # controller_change.parameter1 = 64
    # controller_change.parameter2 = 127
    # track.events.append(controller_change)
    # track.events.append(no_time_spacing)

    return track


def end_track(track):
    track.events.append(DeltaTime(track=track, time=0))

    end_of_track = MidiEvent(track=track, type=midi.MetaEvents.END_OF_TRACK)
    end_of_track.data = b''
    track.events.append(end_of_track)

    return track


def make_midi(quantized_data, ticks_per_sixteenth_note):
    result = MidiFile()
    result.format = 0
    result.ticksPerQuarterNote = ticks_per_sixteenth_note * 4
    track = write_starting_messages(MidiTrack(0))

    beats_passed = 0

    notes_to_end = []

    leading_delta_time_written = False

    last_event_written_at_beat = 0

    for beat in quantized_data:
        to_write_end = find_notes_to_remove(notes_to_end)
        last_event_written_at_beat += int(end_notes(to_write_end, track, (
                    beats_passed - last_event_written_at_beat) * ticks_per_sixteenth_note) / ticks_per_sixteenth_note)

        for note in beat:
            if note[1] > 0:
                time = (beats_passed - last_event_written_at_beat) * ticks_per_sixteenth_note
                spacing = DeltaTime(track=track, time=time)
                track.events.append(spacing)

                last_event_written_at_beat = beats_passed

                note_on = MidiEvent(track=track, type=ChannelVoiceMessages.NOTE_ON)
                note_on.velocity = 50
                note_on.pitch = note[0]
                note_on.channel = 1
                track.events.append(note_on)
                notes_to_end.append(note)

        beats_passed += 1

        # toRemove = []
        #
        # for note in notes_to_end:
        #     if note[1] == 0:
        #         time = (beats_passed - last_event_written_at_beat) * ticks_per_sixteenth_note
        #         spacing = DeltaTime(track=track, time=time)
        #         track.events.append(spacing)
        #
        #         last_event_written_at_beat = beats_passed
        #
        #         note_off = MidiEvent(track=track, type=ChannelVoiceMessages.NOTE_OFF)
        #         note_off.pitch = note[0]
        #         note_off.channel = 1
        #         note_off.parameter2 = 0
        #         toRemove.append(note)
        #
        #     note[1] -= 1
        #     beats_passed += 1
        #
        # for note in toRemove:
        #     if note[1] <= 0:
        #         notes_to_end.remove(note)

    # end_notes(notes_to_end, track, ticks_per_sixteenth_note * 2)
    end_notes(notes_to_end, track, ticks_per_sixteenth_note)

    end_track(track)
    result.tracks.append(track)

    return result


def end_notes(notes_to_end, track, first_delta_time_to_use):
    first = True
    time_written = 0

    for note in notes_to_end:
        if first:
            track.events.append(DeltaTime(track=track, time=first_delta_time_to_use))
            time_written = first_delta_time_to_use
            first = False
        else:
            track.events.append(DeltaTime(track=track, time=0))

        end_note = MidiEvent(track, type=ChannelVoiceMessages.NOTE_OFF)
        end_note.pitch = note[0]
        end_note.channel = 1
        end_note.parameter2 = 0

        track.events.append(end_note)

    return time_written


def find_notes_to_remove(notes_to_end):
    result = []

    for note in notes_to_end:
        if note[1] == 0:
            result.append(note)
        else:
            note[1] -= 1

    for note in result:
        notes_to_end.remove(note)

    return result


def merge_midi(midi):
    merged = merge_tracks(midi.tracks)
    result = MidiFile()
    result.format = 0
    result.tracks = merged

    return result


def find_note_stops(midi):
    for tInedx, track in enumerate(midi.tracks):
        for eIndex, event in enumerate(track.events):
            if event.type == ChannelVoiceMessages.NOTE_OFF:
                print("Note off", tInedx, eIndex)


def read_quantize_write_midi(nameRead, nameWrite):
    read = MidiFile()
    read.open(nameRead)
    read.read()
    read.close()

    find_note_stops(read)

    merged_track = merge_tracks(read.tracks)
    quantized_notes = quantize_midi(merged_track[0], read.ticksPerQuarterNote)

    write = make_midi(quantized_notes, read.ticksPerQuarterNote // 4)
    write.open(nameWrite + " modified.mid", 'wb')
    write.write()
    write.close()

def get_longest_track(midi):
    index = 0
    len = 0

    curIndex = 0
    for track in midi.tracks:
        if track.length > len:
            len = track.length
            index = curIndex

        curIndex += 1

    return midi.tracks[index]


def load_data(path):

    start = timer()
    midi_in = MidiFile()
    midi_in.open(path)
    midi_in.read()

    time_spent[0] += timer() - start

    start = timer()
    merged = merge_tracks(midi_in.tracks)
    time_spent[1] += timer() - start

    start = timer()
    quantized = quantize_midi(merged[0], midi_in.ticksPerQuarterNote)
    midi_in.close()
    time_spent[2] = timer() - start

    return quantized


def load_all_from_a_folder(folder_path):
    #TODO need to change this away from hardcoded
    noteDataset = [-1] * 500
    # lengthDataset = []

    cur_song = 0

    for root, dirs, files in os.walk(folder_path):
        for name in files:
            if '.mid' in name:
                midData = load_data(folder_path + name)

                # filteredBritaWater = []
                #
                # for i in range(0, len(midData)):
                #     if len(midData[i]) != 0:
                #         filteredBritaWater.append(midData[i])

                # notes = convert_quantized_to_input(midData)
                start = timer()
                notes = remove_rhythm(midData)
                time_spent[3] = timer() - start

                # print(max([len(chord) for chord in notes]))

                start = timer()
                noteDataset[cur_song] = notes
                cur_song += 1
                time_spent[4] = timer() - start
                # noteDataset.append(notes)
                # lengthDataset.append(lengths)

    return noteDataset[:cur_song]


# Expects for the rhythm to be removed already and the chord to use the midi number not 0 based number
def convert_chord_to_output(chord):
    result = get_class(0)
    count = 0

    chord = -(np.sort(-chord))

    for note in chord:
        if note == 0:
            break
        count += 1

    median_note = chord[count // 2]

    return median_note


def convert_quantized_to_input(midiData):
    notes = zeroes_output * len(midiData)
    i = 0
    for item in midiData:
        added = False
        for note in item:
            if note[0] >= 21:
                if not added:
                    notes[i] = [0] * num_notes
                    added = True

                notes[i][note[0] - 21] = 1
        i += 1

    return np.array(notes)


chord_dictionary = {}
# Removes rhythm and converts to one hot encoding
def get_chord(stripped_chord):

    chord_tuple = tuple(stripped_chord)
    if chord_tuple in chord_dictionary:
        return chord_dictionary[chord_tuple]
    else:
        chord = np.zeros(num_notes)
        for note in stripped_chord:
            chord[note] = 1
        chord_dictionary[chord_tuple] = chord
        return chord


def remove_rhythm(midiData):
    chord_count = 0
    for chord in midiData:
        chord_count += 1


    #TODO modify this to allow a chord of greater size
    notes = np.zeros((chord_count,16), dtype=np.float32)

    cur_chord = 0
    for chord in midiData:
        i = 0
        for note in chord:
            notes[cur_chord][i] = note[0]
            i += 1
            if i == notes_per_chord:
                break

        cur_chord+=1

    return notes




def load_all_examples(cwd):
    noteDataset = load_all_from_a_folder(cwd + "Music/piano-midi.de/")
    noteDataset += load_all_from_a_folder(cwd + 'Music/kunstderfuge.com/')
    return np.array(noteDataset)


def prepare_examples_for_series_generator(cwd):
    print("Loading midi files from disk")
    noteDataset = load_all_examples(cwd)

    print("Finished loading from disk. Converting to input representation")

    number_of_notes = 65

    zero_pad_length = 200
    print("Allocate array space")
    result_data = {}
    print("Finished allocation")
    song_num = 0


    start = timer()

    for song in noteDataset:
        adjusted_song = convert_to_multihot(song, number_of_notes, zero_pad_length)
        result_data[song_num] = adjusted_song
        song_num += 1

    time_spent[5] = timer() - start

    return result_data


def convert_to_multihot(song, number_of_notes, zero_pad_length):
    song_length = len(song)
    result = np.zeros((song_length + 2 * zero_pad_length, number_of_notes), dtype=np.float32)
    for i in range(len(song)):
        firstNote = True

        # Remember, note zero means no note
        for note in song[i]:
            if note == 0 or note is None:
                if firstNote is True:
                    result[i + zero_pad_length][0] = 1
                break
            result[i + zero_pad_length][int(note)] = 1
            firstNote = False

    return result


def prepare_examples_with_views(number_of_notes_per_input, cwd):
    print("Loading midi files from disk")
    noteDataset = load_all_examples(cwd)

    print("Finished loading from disk. Converting to input representation")

    leading_note_blanks = number_of_notes_per_input - 16

    if leading_note_blanks < 0:
        leading_note_blanks = 0

    inputCount = 0

    for song in noteDataset:
        inputCount += len(song) - 1 - number_of_notes_per_input

    print(inputCount)

    note_inputs = np.empty([inputCount, number_of_notes_per_input, notes_per_chord])
    note_outputs = np.empty([inputCount, num_notes])


    song_count = 0

    input_num = 0

    class_weight = {}

    for i in range(num_notes):
        class_weight[i] = 0

    for song in noteDataset:
        # zero_padding = zeroes_input * (leading_note_blanks - 16)
        # song = np.insert(song, 0, zero_padding, axis=0)

        for i in range(len(song) - 1 - number_of_notes_per_input):
            note_inputs[input_num] = song[i:i + number_of_notes_per_input]
            output_note = convert_chord_to_output(song[i + number_of_notes_per_input])
            note_outputs[input_num] = get_class(output_note)
            class_weight[output_note] += 1
            input_num+=1

        song_count+= 1

        if song_count % 20 == 0:
            print("Completed ", song_count, "songs: ", song_count / len(noteDataset), "%")


    print("Finished input representation")

    return note_inputs, note_outputs, class_weight


# if __name__ == '__main__':
#
#     noteDataset = []
#     lengthDataset = []
#     for root, dirs, files in os.walk('../Music/piano-midi.de'):
#         for name in files:
#             if '.mid' in name:
#                 midData = load_data("../Music/piano-midi.de/" + name)
#                 filteredBritaWater = []
#
#                 for i in range(0, len(midData)):
#                     if len(midData[i]) != 0:
#                         filteredBritaWater.append(midData[i])
#
#                 notes = []
#                 lengths = []
#
#                 for item in filteredBritaWater:
#                     notes.append([note[0] for note in item])
#                     lengths.append([note[1] for note in item])
#
#                 print(max([len(chord) for chord in notes]))
#                 noteDataset.append(notes)
#                 lengthDataset.append(lengths)
#
#     for root, dirs, files in os.walk('../Music/kunstderfuge.com'):
#         for name in files:
#             if '.mid' in name:
#                 midData = load_data("../Music/kunstderfuge.com/" + name)
#                 filteredBritaWater = []
#
#                 for i in range(0, len(midData)):
#                     if len(midData[i]) != 0:
#                         filteredBritaWater.append(midData[i])
#
#                 notes = []
#                 lengths = []
#
#                 for item in filteredBritaWater:
#                     notes.append([note[0] for note in item])
#                     lengths.append([note[1] for note in item])
#
#                 print(max([len(chord) for chord in notes]))
#                 noteDataset.append(notes)
#                 lengthDataset.append(lengths)
#
#     fixedNoteDataSet = []
#     for song in noteDataset:
#         fixedSongSet = []
#         for chord in song:
#             fixedChordSet = []
#             print(chord)
#             while len(chord) != 8:
#                 chord.append(0)
#             fixedSongSet.append(chord)
#         fixedNoteDataSet.append(fixedSongSet)
#
#     Inputs = []
#     Labels = []
#     for i in range(0, len(fixedNoteDataSet)):
#         for j in range(0, len(fixedNoteDataSet[i]) - 9):
#             trainingExample = fixedNoteDataSet[i][j:j + 8]
#             trainingLabel = fixedNoteDataSet[i][j + 8]
#             Inputs.append(trainingExample)
#             Labels.append(trainingLabel)
#
#     for i in range(0, len(Inputs)):
#         print('Training Example {}:'.format(i))
#         print(Inputs[i])
#         print(Labels[i])
#
#     Inputs = np.array(Inputs)
#     Labels = np.array(Labels)
#
#     print(Inputs, Labels)
