from train.models import TonicNet
from torch import cat, multinomial
from torch import cuda, load, device, tensor, zeros
from torch.nn import LogSoftmax
import pickle
import random
from copy import deepcopy

"""
Functions to sample from trained models
"""


def sample_TonicNet_random(load_path, max_tokens=2999, temperature=1.0):

    model = TonicNet(nb_tags=98, z_dim=32, nb_layers=3, nb_rnn_units=256, dropout=0.0)

    try:
        if cuda.is_available():
            model.load_state_dict(load(load_path)['model_state_dict'])
        else:
            model.load_state_dict(load(load_path, map_location=device('cpu'))['model_state_dict'])
        print("loded params from", load_path)
    except:
        raise ImportError(f'No file located at {load_path}, could not load parameters')
    print(model)

    if cuda.is_available():
        model.cuda()

    model.eval()
    model.seq_len = 1
    model.hidden = model.init_hidden()
    model.zero_grad()

    inverse_t = pickle.load(open('tokenisers/inverse_pitch_only.p', mode='rb'))

    seed, pos_dict = __get_seed()

    x = seed
    x_post = x

    inst_conv_dict = {0: 0, 1: 1, 2: 4, 3: 2, 4: 3}
    current_token_dict = {0: '', 1: '', 2: '', 3: '', 4: ''}

    print("")
    print(0)
    print("\t", 0, ":", chord_from_token(x[0][0].item() - 48))

    for i in range(max_tokens):

        if i == 0:
            reset_hidden = True
        else:
            reset_hidden = False

        inst = inst_conv_dict[i % 5]
        psx = pos_dict[inst]

        psx_t = tensor(psx).view(1, 1)

        y_hat = model(x, z=psx_t, sampling=True,
                      reset_hidden=reset_hidden).data.view(-1).div(temperature).exp()
        y = multinomial(y_hat, 1)[0]
        if y.item() == 0:  # EOS token
            print("ending")
            break
        else:
            try:
                token = inverse_t[y.item()]
            except:
                token = chord_from_token(y.item() - 48)

        next_inst = inst_conv_dict[(i + 1) % 5]

        print("")
        print(i + 1)

        if current_token_dict[next_inst] == token:
            pos_dict[next_inst] += 1
        else:
            current_token_dict[next_inst] = token
            pos_dict[next_inst] = 0

        x = y.view(1, 1, 1)
        x_post = cat((x_post, x), dim=1)

    return x_post


def sample_TonicNet_beam_search(load_path, max_tokens=2999, beam_width=10, alpha=1.0):

    """sample the model via beam search algorithm (not the most efficient implementation but functional)

    :param load_path: path to state_dict to load weights from
    :param max_tokens: maximum number of iterations to sample
    :param beam_width: breadth of beam search heuristic
    :param alpha: hyperparamter for length normalisation of beam search - higher value prefers longer sequences
    :return: generated list of token indices
    """

    model = TonicNet(nb_tags=98, z_dim=32, nb_layers=3, nb_rnn_units=256, dropout=0.0)
    logsoftmax = LogSoftmax(dim=0)

    try:
        if cuda.is_available():
            model.load_state_dict(load(load_path)['model_state_dict'])
        else:
            model.load_state_dict(load(load_path, map_location=device('cpu'))['model_state_dict'])
        print("loded params from", load_path)
    except:
        raise ImportError(f'No file located at {load_path}, could not load parameters')
    print(model)

    if cuda.is_available():
        model.cuda()

    model.eval()
    model.seq_len = 1
    model.hidden = model.init_hidden()
    model.zero_grad()

    inverse_t = pickle.load(open('tokenisers/inverse_pitch_only.p', mode='rb'))
    inverse_t[0] = 'end'

    seed, pos_dict = __get_seed()

    x = seed
    x_post = x

    inst_conv_dict = {0: 0, 1: 1, 2: 4, 3: 2, 4: 3}
    current_token_dict = {0: '', 1: '', 2: '', 3: '', 4: ''}

    candidate_seqs = []
    c_ts = []
    pos_ds = []
    models = []
    scores = []

    ended = [0] * beam_width

    for i in range(max_tokens):

        print("")
        print(i)

        log_probs = zeros((98*beam_width))
        updated = [False] * beam_width

        inst = inst_conv_dict[i % 5]
        next_inst = inst_conv_dict[(i + 1) % 5]

        for b in range(beam_width):

            if i == 0:
                reset_hidden = True
                candidate_seqs.append(deepcopy([x]))
                c_ts.append(deepcopy(current_token_dict))
                pos_ds.append(deepcopy(pos_dict))
                models.append(deepcopy(model))
                scores.append(0)
            else:
                reset_hidden = False

            psx = pos_ds[b][inst]

            psx_t = tensor(psx).view(1, 1)

            if candidate_seqs[b][-1].item() == 0:
                log_probs[(98 * b):(98 * (b + 1))] = tensor(98).fill_(-9999)
            else:
                y_hat = models[b](candidate_seqs[b][-1], z=psx_t, sampling=True, reset_hidden=reset_hidden)
                log_probs[(98*b):(98*(b+1))] = logsoftmax(y_hat[0, 0, :]) + scores[b]

        if i == 0:
            top = log_probs[0:98].topk(k=beam_width)
        else:
            top = log_probs.topk(k=beam_width)

        temp_store = []
        rejection_reconciliation = {}

        for b1 in range(beam_width):
            candidate = top[1][b1].item()
            prob = top[0][b1].item()
            y = candidate % 98
            m = int(candidate / 98)

            if updated[m]:
                temp_store.append((m, y, prob))
            else:
                updated[m] = True
                rejection_reconciliation[m] = m
                if candidate_seqs[m][-1].item() is not 0:
                    scores[m] = prob
                    candidate_seqs[m].append(tensor(y).view(1, 1, 1))
                else:
                    ended[m] = 1

        for b2 in range(beam_width):
            if not updated[b2]:
                rejection_reconciliation[b2] = temp_store[0][0]
                if candidate_seqs[b2][-1].item() is not 0:

                    scores[b2] = temp_store[0][2]

                    candidate_seqs[b2] = deepcopy(candidate_seqs[rejection_reconciliation[b2]])
                    candidate_seqs[b2].pop(-1)
                    candidate_seqs[b2].append(tensor(temp_store[0][1]).view(1, 1, 1))

                    models[b2].hidden = models[rejection_reconciliation[b2]].hidden

                    # copy over pos_dict and current token to replace rejected model's
                    pos_dict[b2] = deepcopy(pos_dict[rejection_reconciliation[b2]])
                    c_ts[b2] = deepcopy(c_ts[rejection_reconciliation[b2]])

                    temp_store.pop(0)
                else:
                    ended[b2] = 1

            try:
                token = inverse_t[candidate_seqs[b2][-1].item()]
            except:
                token = chord_from_token(candidate_seqs[b2][-1].item() - 48)

            if c_ts[b2][next_inst] == token:
                pos_ds[b2][next_inst] += 1
            else:
                c_ts[b2][next_inst] = token
                pos_ds[b2][next_inst] = 0

        if sum(ended) == beam_width:
            print('all ended')
            break

    normalised_scores = [scores[n] * (1/(len(candidate_seqs[n])**alpha)) for n in range(beam_width)]

    chosen_seq = max(zip(normalised_scores, range(len(normalised_scores))))[1]

    for n in candidate_seqs[chosen_seq][1:]:
        x_post = cat((x_post, n), dim=1)

    return x_post


def __get_seed():

    seed = random.choice(range(48, 72))
    x = tensor(seed)

    p_dct = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    return x.view(1, 1, 1), p_dct


def chord_from_token(token):

    if token < 12:
        qual = 'major'
    elif token < 24:
        qual = 'minor'
    elif token < 36:
        qual = 'diminished'
    elif token < 48:
        qual = 'augmented'
    elif token == 48:
        qual = 'other'
    else:
        qual = 'none'

    return token % 12, qual

