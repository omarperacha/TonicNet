import music21
import pickle
import matplotlib.pyplot as plt
from preprocessing.utils import get_parts_from_stream

"""
Utility functions for evaluating model training and saving samples from trained models
"""


def indices_to_stream(token_list, return_stream=False):
    inverse_t = pickle.load(open('tokenisers/inverse_pitch_only.p', 'rb'))
    tl = token_list.squeeze()
    tl = tl.numpy()

    sop_part = music21.stream.Part()
    sop_part.id = 'soprano'

    alto_part = music21.stream.Part()
    alto_part.id = 'alto'

    tenor_part = music21.stream.Part()
    tenor_part.id = 'tenor'

    bass_part = music21.stream.Part()
    bass_part.id = 'bass'

    score = music21.stream.Stream([sop_part, bass_part, alto_part, tenor_part])

    for j in range(len(tl)):

        i = tl[j]
        try:
            note = inverse_t[i]
        except:
            continue

        idx = (j % 5) - 1

        if note == 'Rest':
            n = music21.note.Rest()
        else:
            pitch = int(note)
            n = music21.note.Note(pitch)

        dur = 0.25
        n.quarterLength = dur

        score[idx].append(n)

    if return_stream:
        return score
    else:
        score.write('midi', fp='eval/sample.mid')
        print("SAVED sample to ./eval/sample.mid")


def plot_loss_acc_curves(log='eval/out.log'):
    train_loss = []
    train_acc = []

    val_loss = []
    val_acc = []

    f = open(log, "r")
    txt = f.read()
    for line in txt.split("\n"):
        if 'finished' in line:
            components = line.split(" ")
            loss = components[5]
            loss = loss[:-1]
            acc = components[7]
            if 'train phase' in line:
                train_acc.append(float(acc))
                train_loss.append(float(loss))
            else:
                val_acc.append(float(acc))
                val_loss.append(float(loss))

    plt.figure(1)
    plt.subplot(121)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel('epochs')
    plt.legend(['train loss', 'val loss'], loc='upper left')
    plt.ylim(0, 6)

    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.xlabel('epochs')
    plt.legend(['train acc', 'val acc'], loc='upper left')
    plt.ylim(0, 100)
    plt.show()

    plt.show()


def smooth_rhythm():
    path = 'eval/sample.mid'

    mf = music21.midi.MidiFile()
    mf.open(path)
    mf.read()
    mf.close()

    s = music21.midi.translate.midiFileToStream(mf)

    score = music21.stream.Stream()

    parts = get_parts_from_stream(s)

    for part in parts:
        new_part = music21.stream.Part()

        current_pitch = -1
        current_offset = 0.0
        current_dur = 0.0

        for n in part.notesAndRests.flat:
            if isinstance(n, music21.note.Rest):
                if current_pitch == 129:
                    current_dur += 0.25
                else:
                    if current_pitch > -1:
                        if current_pitch < 128:
                            note = music21.note.Note(current_pitch)
                        else:
                            note = music21.note.Rest
                        note.quarterLength = current_dur
                        new_part.insert(current_offset, note)

                        current_pitch = 129
                        current_offset = n.offset
                        current_dur = 0.25

            else:
                if n.pitch.midi == current_pitch:
                    current_dur += 0.25
                else:
                    if current_pitch > -1:
                        if current_pitch < 128:
                            note = music21.note.Note(current_pitch)
                        else:
                            note = music21.note.Rest
                        note.quarterLength = current_dur
                        new_part.insert(current_offset, note)

                    current_pitch = n.pitch.midi
                    current_offset = n.offset
                    current_dur = 0.25

        if current_pitch < 128:
            note = music21.note.Note(current_pitch)
        else:
            note = music21.note.Rest
        note.quarterLength = current_dur
        new_part.insert(current_offset, note)

        score.append(new_part)

    score.write('midi', fp='eval/sample_smoothed.mid')
    print("SAVED rhythmically 'smoothed' sample to ./eval/sample_smoothed.mid")



