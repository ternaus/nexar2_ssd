"""
Annotations for the Nexar 2 challenge are stored in a csv file like:

image_filename,x0,y0,x1,y1,label,confidence
frame_817c47b8-22c4-438a-8dc6-0e3f67f299ee_00000-1280_720.jpg,601.6,270.355731225,726.755555556,421.185770751,van,1.0
frame_817c47b8-22c4-438a-8dc6-0e3f67f299ee_00000-1280_720.jpg,497.777777778,308.774703557,534.755555556,338.656126482,car,1.0
frame_817c47b8-22c4-438a-8dc6-0e3f67f299ee_00000-1280_720.jpg,449.422222222,310.197628458,509.155555556,358.577075099,car,1.0

to access the annotation in an effective manner data is split into separate csv
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from joblib import Parallel, delayed


def save_annotation(x):
    file_name, df = x
    img = Image.open(str(img_path / file_name))

    width, height = img.size

    df.loc[:, 'width'] = width
    df.loc[:, 'height'] = height

    df.loc[:, 'x0'] /= width
    df.loc[:, 'x1'] /= width

    df.loc[:, 'y0'] /= height
    df.loc[:, 'y1'] /= height

    df.to_csv(str(annotation_path / (file_name[:-4] + '.csv')), index=False)


if __name__ == '__main__':
    num_threads = 20

    data_path = Path('data')
    annotation_path = data_path / 'annotations'
    img_path = data_path / 'train'

    annotation_path.mkdir(exist_ok=True)

    labels = pd.read_csv(str(data_path / 'train_boxes.csv'))
    labels['label'] = 'car'  # merging all classes into one

    Parallel(n_jobs=num_threads)(delayed(save_annotation)(x) for x in labels.groupby('image_filename'))
