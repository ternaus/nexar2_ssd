import argparse
from pathlib import Path
import utils.utils
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm
import pandas as pd

from data import AnnotationTransform, NexarDetection, BaseTransform, NEXAR_CLASSES
from ssd import build_ssd

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model',
                    default='weights/ssd512_0712_85000.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval', type=str, help='Dir to save results')
parser.add_argument(
    '--visual_threshold', default=0.01, type=float, help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--root', default='data', help='Location of Nexar root directory')
parser.add_argument('--input_path', default='data/test', help='Location of Nexar test directory')
parser.add_argument('--ssd_type', default=300, type=int, help='type of the SSD 300 or 512')
parser.add_argument('--device_ids', default='0', type=str, help='cuda device ids 0,1,2,3')


args = parser.parse_args()

print('Running with args:')
for arg in vars(args):
    print('    {}: {}'.format(arg, getattr(args, arg)))
print()

Path(args.save_folder).mkdir(exist_ok=True)


def cuda(x):
    return x.cuda() if torch.cuda.is_available else x


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def test_net(save_folder: Path, net, input_path: Path, transform, threshold: float):
    # dump predictions and assoc. ground truth to text file for now

    temp = []

    for file_name in tqdm(sorted(list(input_path.glob('*')))[:10]):
        img = utils.utils.load_image(file_name)

        height, width, _ = img.shape
        scale = torch.Tensor([width, height, width, height])

        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)
        detections = y.data
        # scale each detection back up to the image

        for i in range(detections.size(1) - 1):  # -1 because we want to exclude background
            j = 0
            while detections[0, i, j, 0] >= threshold:
                score = detections[0, i, j, 0]
                label_name = NEXAR_CLASSES[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                temp += [(pt[0], pt[1], pt[2], pt[2], str(file_name.name), label_name, score)]
                j += 1

    df = pd.DataFrame(temp, columns=['x_min', 'y_min', 'x_max', 'y_max', 'class_name', 'score'])
    df.to_csv(str(save_folder / 'preds.csv'), index=False)


if __name__ == '__main__':
    # load net

    num_classes = len(NEXAR_CLASSES) + 1  # +1 background
    net = build_ssd('test', args.ssd_type, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()

    if args.cuda:
        device_ids = [int(x) for x in args.device_ids.split(',')]
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
        cudnn.benchmark = True

    print('Finished loading model!')
    # load data
    root = Path(args.root)

    testset = NexarDetection(root, 'test', transform=None,
                             target_transform=AnnotationTransform(), dataset_name=False)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(
        Path(args.save_folder),
        net,
        Path(args.input_path),
        BaseTransform(args.ssd_type, (104, 117, 123)),
        threshold=args.visual_threshold)
