import torch
import os


def save_model(model, epoch, directory='model_checkpoint'):
    if not os.path.exists(directory):
        os.mkdir(directory)

    checkpoint_name = 'epoch_{}'.format(epoch)
    path = os.path.join(directory, checkpoint_name)
    torch.save(model.state_dict(), path)
    print('Saved model at epoch {} successfully'.format(epoch))
    with open('{}/checkpoint'.format(directory), 'w') as file:
        file.write(checkpoint_name)
        print('Write to checkpoint')


def load_model(model, checkpoint_name=None, directory='model_checkpoint'):
    if checkpoint_name is None:
        with open('{}/checkpoint'.format(directory)) as file:
            content = file.read().strip()
            path = os.path.join(directory, content)
    else:
        path = os.path.join(directory, checkpoint_name)

    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(path))
    return model
