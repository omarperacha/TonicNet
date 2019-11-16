import torch
import os
import pickle
import numpy as np
from preprocessing.instruments import get_instrument
from random import sample

"""
File containing functions to derive training data for neural networks
"""

CUDA = torch.cuda.is_available()

if CUDA:
    PATH = 'train/training_set/X_cuda'
else:
    PATH = 'train/training_set/X'
if os.path.exists(PATH):
    TRAIN_BATCHES = len(os.listdir(PATH))
else:
    TRAIN_BATCHES = 0
TOTAL_BATCHES = TRAIN_BATCHES + 76

MAX_SEQ = 2880
N_PITCH = 48
N_CHORD = 50
N_TOKENS = N_PITCH + N_CHORD


def get_data_set(mode, shuffle_batches=True, return_I=False):

    if mode == 'train':
        parent_dir = 'train/training_set'
    elif mode == 'val':
        parent_dir = 'train/val_set'
    else:
        raise Exception("invalid mode passed to get_data_set() - options are 'train' and 'val'")

    if torch.cuda.is_available():
        lst = os.listdir(f'{parent_dir}/X_cuda')
    else:
        lst = os.listdir(f'{parent_dir}/X')
    try:
        lst.remove('.DS_Store')
    except:
        pass
    
    if shuffle_batches:
        lst = sample(lst, len(lst))

    for file_name in lst:
        if torch.cuda.is_available():
            X = torch.load(f'{parent_dir}/X_cuda/{file_name}')
            Y = torch.load(f'{parent_dir}/Y_cuda/{file_name}')
            P = torch.load(f'{parent_dir}/P_cuda/{file_name}')
            if return_I:
                I = torch.load(f'{parent_dir}/I_cuda/{file_name}')
                C = torch.load(f'{parent_dir}/C_cuda/{file_name}')
        else:
            X = torch.load(f'{parent_dir}/X/{file_name}')
            Y = torch.load(f'{parent_dir}/Y/{file_name}')
            P = torch.load(f'{parent_dir}/P/{file_name}')
            if return_I:
                I = torch.load(f'{parent_dir}/I/{file_name}')
                C = torch.load(f'{parent_dir}/C/{file_name}')

        if return_I:
            yield X, Y, P, I, C
        else:
            yield X, Y, P


def bach_chorales_classic(mode, transpose=False, maj_min=False):

    tokeniser = pickle.load(open('tokenisers/pitch_only.p', 'rb'))
    tokeniser["end"] = 0
    count = 0

    for folder_name in ["training_set", "val_set"]:
        if torch.cuda.is_available():
            print("cuda:")
            try:
                os.makedirs(f'train/{folder_name}/X_cuda')
                os.makedirs(f'train/{folder_name}/Y_cuda')
                os.makedirs(f'train/{folder_name}/P_cuda')
                os.makedirs(f'train/{folder_name}/I_cuda')
                os.makedirs(f'train/{folder_name}/C_cuda')
            except:
                pass
        else:
            try:
                os.makedirs(f'train/{folder_name}/X')
                os.makedirs(f'train/{folder_name}/Y')
                os.makedirs(f'train/{folder_name}/P')
                os.makedirs(f'train/{folder_name}/I')
                os.makedirs(f'train/{folder_name}/C')
            except:
                pass

    for phase in ['train', 'valid']:

        d = np.load('dataset_unprocessed/Jsb16thSeparated.npz', allow_pickle=True, encoding="latin1")
        train = (d[phase])

        ks = pickle.load(open(f'dataset_unprocessed/{phase}_keysigs.p', 'rb'))
        crds = pickle.load(open(f'dataset_unprocessed/{phase}_chords.p', 'rb'))
        crds_majmin = pickle.load(open('dataset_unprocessed/train_majmin_chords.p', 'rb'))
        k_count = 0

        for m in train:
            int_m = m.astype(int)

            tonic = ks[k_count][0]
            scale = ks[k_count][1]
            crd = crds[k_count]
            crd_majmin = crds_majmin[k_count]
            k_count += 1

            if transpose is False or phase == 'valid':
                transpositions = [int_m]
                crds_pieces = [crd]
            else:
                parts = [int_m[:, 0], int_m[:, 1], int_m[:, 2], int_m[:, 3]]
                transpositions, tonics, crds_pieces = __np_perform_all_transpositions(parts, tonic, crd)

                if maj_min:

                    mode_switch = __np_convert_major_minor(int_m, tonic, scale)
                    ms_parts = [mode_switch[:, 0], mode_switch[:, 1], mode_switch[:, 2], mode_switch[:, 3]]
                    ms_trans, ms_tons, ms_crds = __np_perform_all_transpositions(ms_parts, tonic, crd_majmin)

                    transpositions += ms_trans
                    tonics += ms_tons
                    crds_pieces += ms_crds

            kc = 0

            for t in transpositions:

                crds_piece = crds_pieces[kc]

                _tokens = []
                inst_ids = []
                c_class = []

                current_s = ''
                s_count = 0

                current_a = ''
                a_count = 0

                current_t = ''
                t_count = 0

                current_b = ''
                b_count = 0

                current_c = ''
                c_count = 0

                timestep = 0

                for i in t:
                    s = 'Rest' if i[0] < 36 else str(i[0])
                    b = 'Rest' if i[3] < 36 else str(i[3])
                    a = 'Rest' if i[1] < 36 else str(i[1])
                    t = 'Rest' if i[2] < 36 else str(i[2])

                    c_val = crds_piece[timestep] + 48
                    timestep += 1

                    _tokens = _tokens + [c_val, s, b, a, t]
                    c_class = c_class + [c_val]

                    if c_val == current_c:
                        c_count += 1
                    else:
                        c_count = 0
                        current_c = c_val

                    if s == current_s:
                        s_count += 1
                    else:
                        s_count = 0
                        current_s = s

                    if b == current_b:
                        b_count += 1
                    else:
                        b_count = 0
                        current_b = b

                    if a == current_a:
                        a_count += 1
                    else:
                        a_count = 0
                        current_a = a

                    if t == current_t:
                        t_count += 1
                    else:
                        t_count = 0
                        current_t = t

                    inst_ids = inst_ids + [c_count, s_count, b_count, a_count, t_count]

                pos_ids = list(range(len(_tokens)))

                kc += 1
                _tokens.append('end')
                tokens = []
                try:
                    for x in _tokens:
                        if isinstance(x, str):
                            tokens.append(tokeniser[x])
                        else:
                            tokens.append(x)
                except:
                    print("ERROR: tokenisation")
                    continue

                SEQ_LEN = len(tokens) - 1

                count += 1

                data_x = []
                data_y = []

                pos_x = []

                for i in range(0, len(tokens) - SEQ_LEN, 1):
                    t_seq_in = tokens[i:i + SEQ_LEN]
                    t_seq_out = tokens[i + 1: i + 1 + SEQ_LEN]
                    data_x.append(t_seq_in)
                    data_y.append(t_seq_out)

                    p_seq_in = pos_ids[i:i + SEQ_LEN]
                    pos_x.append(p_seq_in)

                X = torch.tensor(data_x)
                X = torch.unsqueeze(X, 2)

                Y = torch.tensor(data_y)
                P = torch.tensor(pos_x)
                I = torch.tensor(inst_ids)
                C = torch.tensor(c_class)

                set_folder = 'training_set'
                if phase == 'valid':
                    set_folder = 'val_set'

                if mode == 'save':

                    if torch.cuda.is_available():
                        print("cuda:")
                        torch.save(X.cuda(), f'train/{set_folder}/X_cuda/{count}.pt')
                        torch.save(Y.cuda(), f'train/{set_folder}/Y_cuda/{count}.pt')
                        torch.save(P.cuda(), f'train/{set_folder}/P_cuda/{count}.pt')
                        torch.save(I.cuda(), f'train/{set_folder}/I_cuda/{count}.pt')
                        torch.save(C.cuda(), f'train/{set_folder}/C_cuda/{count}.pt')
                    else:
                        torch.save(X, f'train/{set_folder}/X/{count}.pt')
                        torch.save(Y, f'train/{set_folder}/Y/{count}.pt')
                        torch.save(P, f'train/{set_folder}/P/{count}.pt')
                        torch.save(I, f'train/{set_folder}/I/{count}.pt')
                        torch.save(C, f'train/{set_folder}/C/{count}.pt')
                    print("saved", count)
                else:
                    print("processed", count)
                    yield X, Y, P, I, C


def get_test_set_for_eval_classic(phase='test'):

    tokeniser = pickle.load(open('tokenisers/pitch_only.p', 'rb'))
    tokeniser["end"] = 0

    d = np.load('dataset_unprocessed/Jsb16thSeparated.npz', allow_pickle=True, encoding="latin1")
    test = (d[f'{phase}'])

    crds = pickle.load(open(f'dataset_unprocessed/{phase}_chords.p', 'rb'))
    crd_count = 0

    for m in test:
        int_m = m.astype(int)

        crds_piece = crds[crd_count]
        crd_count += 1

        _tokens = []
        inst_ids = []
        c_class = []

        current_s = ''
        s_count = 0

        current_a = ''
        a_count = 0

        current_t = ''
        t_count = 0

        current_b = ''
        b_count = 0

        current_c = ''
        c_count = 0

        timestep = 0

        for i in int_m:
            s = 'Rest' if i[0] < 36 else str(i[0])
            b = 'Rest' if i[3] < 36 else str(i[3])
            a = 'Rest' if i[1] < 36 else str(i[1])
            t = 'Rest' if i[2] < 36 else str(i[2])

            c_val = crds_piece[timestep] + 48
            timestep += 1

            _tokens = _tokens + [c_val, s, b, a, t]
            c_class = c_class + [c_val]

            if c_val == current_c:
                c_count += 1
            else:
                c_count = 0
                current_c = c_val

            if s == current_s:
                s_count += 1
            else:
                s_count = 0
                current_s = s

            if b == current_b:
                b_count += 1
            else:
                b_count = 0
                current_b = b

            if a == current_a:
                a_count += 1
            else:
                a_count = 0
                current_a = a

            if t == current_t:
                t_count += 1
            else:
                t_count = 0
                current_t = t

            inst_ids = inst_ids + [c_count, s_count, b_count, a_count, t_count]

        pos_ids = list(range(len(_tokens)))

        _tokens.append('end')

        tokens = []
        try:
            for x in _tokens:
                if isinstance(x, str):
                    tokens.append(tokeniser[x])
                else:
                    tokens.append(x)
        except:
            print("ERROR: tokenisation")
            continue

        SEQ_LEN = len(tokens) - 1

        data_x = []
        data_y = []

        pos_x = []

        for i in range(0, len(tokens) - SEQ_LEN, 1):
            t_seq_in = tokens[i:i + SEQ_LEN]
            t_seq_out = tokens[i + 1: i + 1 + SEQ_LEN]
            data_x.append(t_seq_in)
            data_y.append(t_seq_out)

            p_seq_in = pos_ids[i:i + SEQ_LEN]
            pos_x.append(p_seq_in)

        X = torch.tensor(data_x)
        X = torch.unsqueeze(X, 2)

        Y = torch.tensor(data_y)
        P = torch.tensor(pos_x)
        C = torch.tensor(c_class)
        I = torch.tensor(inst_ids)

        yield X, Y, P, I, C


def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = torch.tensor([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)], dtype=torch.float32)

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim

    if torch.cuda.is_available():
        position_enc = position_enc.cuda()

    return position_enc


def to_categorical(y, num_classes=N_TOKENS):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y]


def __pos_from_beatStr(measure, beat, div=8):

    if len(beat.split(" ")) == 1:
        b_pos = float(beat.split(" ")[0])
    else:
        fraction = beat.split(" ")[1]
        fraction = fraction.split("/")
        decimal = float(fraction[0])/float(fraction[1])
        b_pos = float(beat.split(" ")[0]) + decimal

    b_pos *= div
    b_pos -= div

    pos_enc_idx = int(measure*(4*div) + b_pos)

    return pos_enc_idx


# Mark:- transposition
def __np_perform_all_transpositions(parts, tonic, chords):
    mylist = []
    tonics = []
    my_chords = []
    try:
        t_r = __np_transposable_range_for_piece(parts)
    except:
        print("error getting transpose range")
        return mylist
    lower = t_r[0]
    higher = t_r[1] + 1
    quals = [__np_get_quals_from_chord(x) for x in chords]
    for i in range(lower, higher):
        try:
            roots = [(x + 12 + i) % 12 for x in chords]
            transposed_piece = np.zeros((len(parts[0]), 4), dtype=int)
            chord_prog = [__np_chord_from_root_qual(roots[i], quals[i]) for i in range(len(chords))]
            for j in range(4):
                tp = parts[j] + i
                transposed_piece[:, j] = tp[:]
        except:
            print("ERROR: empty return")
        else:
            mylist.append(transposed_piece)
            tonics.append((tonic + i) % 12)
            my_chords.append(chord_prog)
    return mylist, tonics, my_chords


def __np_transposable_range_for_part(part, inst):

    if not isinstance(inst, str):
        inst = str(inst)
    part_range = __np_get_part_range(part)
    instrument = get_instrument(inst)

    lower_transposable = instrument.lowestNote - part_range[0]
    higher_transposable = instrument.highestNote - part_range[1]

    # suggests there's perhaps no musical content in this score
    if higher_transposable - lower_transposable >= 128:
        lower_transposable = 0
        higher_transposable = 0
    return min(0, lower_transposable), max(0, higher_transposable)


def __np_transposable_range_for_piece(parts):

    insts = ['soprano', 'alto', 'tenor', 'bass']

    lower = -127
    higher = 127

    for i in range(len(parts)):
        t_r = __np_transposable_range_for_part(parts[i], insts[i])
        if t_r[0] > lower:
            lower = t_r[0]
        if t_r[1] < higher:
            higher = t_r[1]
    # suggests there's perhaps no musical content in this score
    if higher - lower >= 128:
        lower = 0
        higher = 0
    return lower, higher


def __np_get_part_range(part):

    mn = min(part)

    if mn < 36:
        p = sorted(part)
        c = 1
        while mn < 36:
            mn = p[c]
            c += 1

    return [mn, max(part)]


def __np_convert_major_minor(piece, tonic, mode):

    _piece = piece

    for i in range(len(_piece)):
        s = _piece[i][0] if _piece[i][0] < 36 else (_piece[i][0] - tonic) % 12
        b = _piece[i][3] if _piece[i][3] < 36 else (_piece[i][3] - tonic) % 12
        a = _piece[i][1] if _piece[i][1] < 36 else (_piece[i][1] - tonic) % 12
        t = _piece[i][2] if _piece[i][2] < 36 else (_piece[i][2] - tonic) % 12

        parts = [s, a, t, b]

        for n in range(len(parts)):
            if mode == 'major':
                if parts[n] in [4, 9]:
                    _piece[i][n] -= 1
            elif mode == 'minor':
                if parts[n] in [3, 8, 10]:
                    _piece[i][n] += 1
            else:
                raise ValueError(f"mode must be minor or major, received {mode}")

    return _piece


def __np_get_quals_from_chord(chord):

    if chord < 12:
        qual = 'major'
    elif chord < 24:
        qual = 'minor'
    elif chord < 36:
        qual = 'diminished'
    elif chord < 48:
        qual = 'augmented'
    elif chord == 48:
        qual = 'other'
    else:
        qual = 'none'

    return qual


def __np_chord_from_root_qual(root, qual):

    if qual == "major":
        chord = root
    elif qual == "minor":
        chord = root + 12
    elif qual == "diminished":
        chord = root + 24
    elif qual == "augmented":
        chord = root + 36
    elif qual == "other":
        chord = 48
    elif qual == "none":
        chord = 49

    return chord




