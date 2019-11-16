import sys
from preprocessing.nn_dataset import bach_chorales_classic
from train.train_nn import train_TonicNet, TonicNet_lr_finder, TonicNet_sanity_test
from train.train_nn import CrossEntropyTimeDistributedLoss
from train.models import TonicNet
from eval.utils import plot_loss_acc_curves, indices_to_stream, smooth_rhythm
from eval.eval import eval_on_test_set
from eval.sample import sample_TonicNet_random

if len(sys.argv) > 1:
    if sys.argv[1] in ['--train', '-t']:
        train_TonicNet(3000, shuffle_batches=1, train_emb_freq=1, load_path='')

    elif sys.argv[1] in ['--plot', '-p']:
        plot_loss_acc_curves()

    elif sys.argv[1] in ['--find_lr', '-lr']:
        TonicNet_lr_finder(train_emb_freq=1, load_path='')

    elif sys.argv[1] in ['--sanity_test', '-st']:
        TonicNet_sanity_test(num_batches=1, train_emb_freq=1)

    elif sys.argv[1] in ['--sample', '-s']:
        x = sample_TonicNet_random(load_path='eval/TonicNet_epoch-56_loss-0.328_acc-90.750.pt', temperature=1.0)
        indices_to_stream(x)
        smooth_rhythm()

    elif sys.argv[1] in ['--eval_nn', '-e']:
        eval_on_test_set(
            'eval/TonicNet_epoch-58_loss-0.317_acc-90.928.pt',
            TonicNet(nb_tags=98, z_dim=32, nb_layers=3, nb_rnn_units=256, dropout=0.0),
            CrossEntropyTimeDistributedLoss(), set='test', notes_only=True)

    elif sys.argv[1] in ['--gen_dataset', '-gd']:
        for x, y, p, i, c in bach_chorales_classic('save', transpose=True):
            continue

    else:
        print("")
        print("TonicNet (Training on Ordered Notation Including Chords)")
        print("Omar Peracha, 2019")
        print("")
        print("--gen_dataset\t\t prepare dataset")
        print("--train\t\t\t train model from scratch")
        print("--eval_nn\t\t evaluate pretrained model on test set")
        print("--sample\t\t sample from pretrained model")
        print("")
else:

    print("")
    print("TonicNet (Training on Ordered Notation Including Chords)")
    print("Omar Peracha, 2019")
    print("")
    print("--gen_dataset\t\t prepare dataset")
    print("--train\t\t\t train model from scratch")
    print("--eval_nn\t\t evaluate pretrained model on test set")
    print("--sample\t\t sample from pretrained model")
    print("")





