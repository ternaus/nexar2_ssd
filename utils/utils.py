from torchvision.transforms import ToTensor, Normalize, Compose
import cv2


def load_image(path):
    """

    :param path:
    :return: image in RGB format
    """
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
