import music21
import pickle

"""
Contains utility functions for dataset preprocessing
"""


def get_parts_from_stream(piece):
    parts = []
    for i in piece:
        if isinstance(i, music21.stream.Part):
            parts.append(i)
    return parts


def pitch_tokeniser_maker():
    post = {"end": 0}
    for i in range(36, 82):
        k = str(i)
        post[k] = len(post)
    post['Rest'] = len(post)

    return post


def load_tokeniser():
    dic = pickle.load(open("tokenisers/pitch_only.p", "rb"))
    return dic


def chord_from_pitches(pitches):
    cd = []
    for n in pitches:
        if n >= 36:
            cd.append(int(n))

    crd = music21.chord.Chord(cd)
    try:
        root = music21.pitch.Pitch(crd.root()).pitchClass
    except:
        c_val = 49
    else:
        if crd.quality == 'major':
            c_val = root
        if crd.quality == 'minor':
            c_val = root + 12
        if crd.quality == 'diminished':
            c_val = root + 24
        if crd.quality == 'augmented':
            c_val = root + 36
        if crd.quality == 'other':
            c_val = 48

    return c_val

