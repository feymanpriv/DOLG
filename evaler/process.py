#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Written by feymanpriv

import cv2
import numpy as np

_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]


def preprocess(im, scale_factor):
    im = im_scale(im, scale_factor) 
    im = im.transpose([2, 0, 1])
    im = im / 255.0
    im = color_norm(im, _MEAN, _SD)
    return im


def im_scale(im, scale_factor):
    h, w = im.shape[:2]
    h_new = int(round(h * scale_factor))
    w_new = int(round(w * scale_factor))
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.float32)


def color_norm(im, mean, std):
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    img = img.astype(np.float32, copy=False)
    img = preprocess(img.copy(), 1.0)
    print(img.shape[:2]) 
