"""
File containing 'structs' and methods pertaining to determining and assigning instruments to parts in corpus
"""


# MARK:- Instrument Data Objects

class SopranoVoice:
    instrumentId = 'soprano'
    highestNote = 81
    lowestNote = 60


class AltoVoice:
    instrumentId = 'alto'
    highestNote = 77
    lowestNote = 53


class TenorVoice:
    instrumentId = 'tenor'
    highestNote = 72
    lowestNote = 45


class BassVoice:
    instrumentId = 'bass'
    highestNote = 64
    lowestNote = 36


def get_instrument(inst_name_in):
    """

    :param inst_name_in: string with the name of an instrument
    :return: a data object corresponding to the inst_name if applicable, or UNK
    """

    if not isinstance(inst_name_in, str):
        inst_name = str(inst_name_in)
    else:
        inst_name = inst_name_in

    # Handle a few scenarios where multiple instruments could be scored
    if 'bass' in inst_name.lower() or 'B.' in inst_name:
        return BassVoice()

    elif 'tenor' in inst_name.lower():
        return TenorVoice()

    elif 'alto' in inst_name.lower():
        return AltoVoice()

    elif 'soprano' in inst_name.lower() or 'S.' in inst_name:
        return SopranoVoice()

    elif 'canto' in inst_name.lower():
        return SopranoVoice()


def get_part_range(part):
    notes = part.pitches
    midi = list(map(__get_midi, notes))
    return [min(midi), max(midi)]


def __get_midi(pitch):
    return pitch.midi

