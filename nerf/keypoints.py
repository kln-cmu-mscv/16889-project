import cv2
import torch
import numpy as np

from render_functions import get_device

class KeyPoints:

    KEYPOINTS_NAME_TO_I = {'not-keypoint': 0,
                           'bottom-back-left': 1,
                           'bottom-back-right': 2,
                           'bottom-front-left': 3,
                           'bottom-front-right': 4,
                           'cont-bottom-left': 5,
                           'cont-bottom-right': 6,
                           'cont-top-left': 7,
                           'cont-top-right': 8,
                           'top-back-left': 9,
                           'top-back-right': 10,
                           'top-front-left': 11,
                           'top-front-right': 12}

    KEYPOINTS_I_TO_NAME = {0: 'not-keypoint',
                           1: 'bottom-back-left',
                           2: 'bottom-back-right',
                           3: 'bottom-front-left',
                           4: 'bottom-front-right',
                           5: 'cont-bottom-left',
                           6: 'cont-bottom-right',
                           7: 'cont-top-left',
                           8: 'cont-top-right',
                           9: 'top-back-left',
                           10: 'top-back-right',
                           11: 'top-front-left',
                           12: 'top-front-right'}

    KEYPOINTS_COLORS = [(230, 25, 75),
                        (60, 180, 75),
                        (255, 225, 25),
                        (0, 130, 200),
                        (245, 130, 48),
                        (145, 30, 180),
                        (70, 240, 240),
                        (240, 50, 230),
                        (210, 245, 60),
                        (250, 190, 212),
                        (0, 128, 128),
                        (220, 190, 255),
                        (170, 110, 40),
                        (255, 250, 200),
                        (128, 0, 0),
                        (170, 255, 195),
                        (128, 128, 0),
                        (255, 215, 180),
                        (0, 0, 128),
                        (128, 128, 128),
                        (255, 255, 255),
                        (0, 0, 0)]

    def __init__(self, keypoint_data_path='data/keypoints.csv'):

        # Data - (label, x, y, image_ind, W, H)
        raw_data = np.loadtxt(keypoint_data_path, delimiter=',', dtype='U')
        img_indices = np.unique([int(i.split('.')[0]) for i in raw_data[:, 3]])
        W = int(raw_data[0, 4])
        H = int(raw_data[0, 5])

        keypoints_yx = {i: {} for i in img_indices}

        for i in range(raw_data.shape[0]):

            ind = int(raw_data[i, 3].split('.')[0])
            keypoints_yx[ind][self.KEYPOINTS_NAME_TO_I[raw_data[i, 0]]] = \
                [float(raw_data[i, 2]) / H, float(raw_data[i, 1]) / W]

        # This is stored as a dictionary of image indices to dictionary of
        # labels: [y, x]
        self.keypoints = keypoints_yx
        self.image_indices = img_indices
        self.device = get_device()

    def add_keypoint_channel(self, images):

        # Images - (N, h, w, 3)
        # Returns - {i: (h, w, 4)}
        # assert images.size(0) == self.image_indices.shape[0]

        w = images.size(2)
        h = images.size(1)

        images_with_channels = []

        for k in self.keypoints.keys():

            channel = torch.ones((h, w, 1)) * \
                self.KEYPOINTS_NAME_TO_I['not-keypoint']

            neighbors = [(-1, 0),
                         (-1, 0),
                         (-1, 1),
                         (0, -1),
                         (0, 1),
                         (1, -1),
                         (1, 0),
                         (1, 1)]

            for l, coords in self.keypoints[k].items():
                channel[int(coords[0] * h) - 1, int(coords[1] * w) - 1] = l

                for dx, dy in neighbors:
                    if int(coords[0] * h) - 1 + dy >= h  or \
                       int(coords[1] * w) - 1 + dx >= w:
                       continue 
                
                    channel[int(coords[0] * h) - 1 + dy,
                            int(coords[1] * w) - 1 + dx] = l

            images_with_channels.append(torch.dstack((images[k], channel)))

        images_with_channels = torch.stack(images_with_channels).to(self.device)

        images_with_channels = {i: img for (i, img) in zip(self.keypoints.keys(),
                                                           images_with_channels)}

        print('keypoints', 'image_with_channels_0', images_with_channels[0].shape)

        return images_with_channels
