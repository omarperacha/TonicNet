from torch import cuda, load, device, set_grad_enabled, max, sum, cat
from preprocessing.nn_dataset import get_test_set_for_eval_classic

"""
File containing functions to quantitatively evaluate trained models
"""


def eval_on_test_set(load_path, model, criterion, set='test', notes_only=False):
    model = model

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
    criterion = criterion

    count = 0
    batch_count = 0
    loss_epoch = 0
    running_accuray = 0.0
    running_batch_count = 0
    print_loss_batch = 0  # Reset on print
    print_acc_batch = 0  # Reset on print
    pr_interval = 1

    for x, y, psx, i, c in get_test_set_for_eval_classic(set):
        model.zero_grad()

        train_emb = False

        Y = y

        with set_grad_enabled(False):
            y_hat = model(x, z=i, train_embedding=train_emb)
            _, preds = max(y_hat, 2)

            if notes_only:
                for j in range(y_hat.shape[1]):
                    if j % 5 != 4:
                        if j == 0:
                            new_y_hat = y_hat[:, j, :].view(1, 1, 98)
                            new_y = Y[:, j].view(1, 1)
                        else:
                            new_y_hat = cat((new_y_hat, y_hat[:, j, :].view(1, 1, 98)), dim=1)
                            new_y = cat((new_y, Y[:, j].view(1, 1)), dim=1)
                loss = criterion(new_y_hat, new_y, )
            else:
                loss = criterion(y_hat, Y, )

        loss_epoch += loss.item()
        print_loss_batch += loss.item()
        if notes_only:
            _, new_preds = max(new_y_hat, 2)
            running_accuray += sum(new_preds == new_y)
            print_acc_batch += sum(new_preds == new_y)
        else:
            running_accuray += sum(preds == Y)
            print_acc_batch += sum(preds == Y)

        count += 1
        if notes_only:
            batch_count += int(x.shape[1] * 0.8)
            running_batch_count += int(x.shape[1] * 0.8)
        else:
            batch_count += x.shape[1]
            running_batch_count += x.shape[1]

        # print loss for recent set of batches
        if count % pr_interval == 0:
            ave_loss = print_loss_batch / pr_interval
            ave_acc = 100 * print_acc_batch.float() / running_batch_count
            print_acc_batch = 0
            running_batch_count = 0
            print('\t\t[%d] loss: %.3f, acc: %.3f' % (count, ave_loss, ave_acc))
            print_loss_batch = 0

    # calculate loss and accuracy for phase
    ave_loss_epoch = loss_epoch / count
    epoch_acc = 100 * running_accuray.float() / batch_count
    print('\tfinished %s phase loss: %.3f, acc: %.3f' % ('eval', ave_loss_epoch, epoch_acc))

