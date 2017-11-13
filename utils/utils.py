import cv2
import torch
from torch.autograd import Variable


def load_image(path):
    """

    :param path:
    :return: image in RGB format
    """
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cuda(x):
    return x.cuda() if torch.cuda.is_available else x


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))
