"""
This file provides codes of image decomposition and assemble
An image is decomposed into overlapping patches based on sliding cutting
The image could be re-assembled from those patches.

without specific notation, bbox denotes [x, y, w, h]
bbox_xyxy denotes [x1, y1, x2, y2]
where x=x1, y=y1, w=x2-x1, h=y2-y1
and maybe cx=(x1+x2)/2, cy=(y1+y2)/2
"""
import math
import numpy as np
import cv2
from collections import defaultdict, namedtuple

import copy
from pycocotools import mask as maskUtils

Patch = namedtuple('Patch', ['image', 'annos'])


def pad_image(input, pad, mode='constant', value=0):
    """

    :param input: input image
    :param pad: list, [top, bottom, left, right]
    :param mode: currently 'constant' only
    :param value:
    :return: image
    """
    assert len(pad) % 2 == 0, 'Padding length must be divisible by 2'
    assert len(pad) // 2 <= len(input.shape), 'Padding length too large'

    if mode == 'constant':
        output = cv2.copyMakeBorder(input, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=value)
    else:
        raise NotImplementedError()

    return output


def compute_overlapping_box(bbox1, bbox2, type='xyxy'):
    maxx1, maxy1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
    if type == 'xyxy':
        minx2, miny2 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        if maxx1 >= minx2 or maxy1 >= miny2:
            return []
        else:
            return [maxx1, maxy1, minx2, miny2]
    elif type == 'xywh':
        minx2, miny2 = min(bbox1[2] + bbox1[0], bbox2[2] + bbox2[0]), min(bbox1[3] + bbox1[1], bbox2[3] + bbox2[1])
        if maxx1 >= minx2 or maxy1 >= miny2:
            return []
        else:
            return [maxx1, maxy1, minx2 - maxx1, miny2 - maxy1]
    else:
        raise NotImplementedError(f'{type} not recognized.')


def decompose_image(image, annos, patch_size, stride):
    """

    :param image: input image
    :param annos:
    :param patch_size: size of patches
    :param stride: overlapping ratio between patches
    :return: a list of image patches
    """
    img_h, img_w, img_d = image.shape

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    else:
        assert isinstance(patch_size, list) or \
               isinstance(patch_size, tuple) or \
               isinstance(patch_size, np.dtype)
    if isinstance(stride, int):
        stride = (stride, stride)
    else:
        assert isinstance(stride, list) or \
               isinstance(stride, tuple) or \
               isinstance(stride, np.dtype)

    num_rows = math.ceil((img_h - patch_size[0]) / stride[0]) + 1
    num_cols = math.ceil((img_w - patch_size[1]) / stride[1]) + 1

    pad_h = (num_rows - 1) * stride[0] + patch_size[0] - img_h
    pad_w = (num_cols - 1) * stride[1] + patch_size[1] - img_w

    image = pad_image(image, [0, pad_h, 0, pad_w])

    patches = defaultdict(Patch)
    patch_cnt = 0
    for i in range(1, num_rows + 1):
        for j in range(1, num_cols + 1):
            up_left_x = stride[1] * (j - 1)
            up_left_y = stride[0] * (i - 1)
            down_right_x = up_left_x + patch_size[1] - 1
            down_right_y = up_left_y + patch_size[0] - 1
            cut_patch = image[up_left_y:down_right_y + 1, up_left_x:down_right_x + 1]

            if annos is not None:
                objects = []
                for _obj in annos['objects']:
                    new_obj = copy.deepcopy(_obj)
                    exterior = new_obj['points']['exterior']
                    bbox_xyxy = [exterior[0][0],
                            exterior[0][1],
                            exterior[1][0],
                            exterior[1][1]]
                    bbox_xyxy = [min(bbox_xyxy[0], bbox_xyxy[2]), min(bbox_xyxy[1], bbox_xyxy[3]), max(bbox_xyxy[0], bbox_xyxy[2]), max(bbox_xyxy[1], bbox_xyxy[3])]
                    bbox1 = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]]
                    assert bbox1 == _obj['bbox'], (bbox1, _obj['bbox'])
                    overlapping_bbox = compute_overlapping_box([up_left_x, up_left_y, down_right_x, down_right_y],
                                                               bbox_xyxy)

                    if overlapping_bbox:
                        new_exterior = [[overlapping_bbox[0] - up_left_x, overlapping_bbox[1] - up_left_y],
                                        [overlapping_bbox[2] - up_left_x, overlapping_bbox[3] - up_left_y]]
                        new_obj['points']['exterior'] = new_exterior
                        new_obj['bbox'] = [new_exterior[0][0],
                                           new_exterior[0][1],
                                           new_exterior[1][0] - new_exterior[0][0],
                                           new_exterior[1][1] - new_exterior[0][1]]

                        segmentation = new_obj['mask']
                        m = maskUtils.decode(segmentation)
                        m = pad_image(m, [0, pad_h, 0, pad_w])
                        m_patch = m[up_left_y:down_right_y + 1, up_left_x:down_right_x + 1]
                        seg_patch = maskUtils.encode(np.asfortranarray(m_patch))
                        seg_patch['counts'] = seg_patch['counts'].decode('utf-8')
                        new_obj['mask'] = seg_patch

                        objects.append(new_obj)
                patch_annos = {'description': annos['description'],
                               'tags': annos['tags'],
                               'size': {'height': patch_size[0], 'width': patch_size[1]},
                               'objects': objects}
                patches[patch_cnt] = Patch(image=cut_patch.copy(), annos=patch_annos.copy())
            else:
                patches[patch_cnt] = Patch(image=cut_patch.copy(), annos=None)
            patch_cnt += 1

    return patches


def assemble_patches(bboxes, patch_size, stride, image_size=(1080, 1920), type='xyxy'):
    new_bboxes = []
    # num_rows = math.ceil((image_size[0] - patch_size[0]) / stride[0]) + 1
    num_cols = math.ceil((image_size[1] - patch_size[1]) / stride[1]) + 1
    if type == 'xyxy':
        for patch_cnt, patch_bboxes in enumerate(bboxes):
            top_left_x = (patch_cnt % num_cols) * stride[1]
            top_left_y = (patch_cnt // num_cols) * stride[0]
            new_bboxes.append(
                [bbox[0] + top_left_x, bbox[1] + top_left_y, bbox[2] + top_left_x, bbox[3] + top_left_y] for bbox in
                patch_bboxes)

    elif type == 'xywh':
        for patch_cnt, patch_bboxes in enumerate(bboxes):
            top_left_x = (patch_cnt % num_cols) * stride[1]
            top_left_y = (patch_cnt // num_cols) * stride[0]
            new_bboxes.append([bbox[0] + top_left_x, bbox[1] + top_left_y, bbox[2], bbox[3]] for bbox in patch_bboxes)
    else:
        raise NotImplementedError(f'{type} not implemented.')

    return new_bboxes

