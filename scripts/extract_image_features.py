#!/usr/bin/env python

"""
Extract Image Features and store them into h5py file.
"""

import os.path
import numpy as np
import tensorflow as tf

from videocap.util import log, imread, imcrop_and_resize
from glob import glob
import h5py

__PATH__ = os.path.abspath(os.path.dirname(__file__))

import vgg19


def extract_features(args):
    batch_size = args.batch_size
    assert batch_size > 10

    image_dir = os.path.join(__PATH__, '../dataset/mscoco/images', args.dataset)
    log.info('Image directory : %s', image_dir)
    assert os.path.exists(image_dir)

    image_files = list(sorted(glob(image_dir + '/*.jpg')))
    N = len(image_files)
    log.info('Total %d images', N)
    assert N > 0

    feature_h5_path = os.path.join(__PATH__, '../dataset/mscoco/features/%s.%s.h5' % (args.dataset, args.network))
    log.info('Target h5 : %s', feature_h5_path)
    fp = h5py.File(feature_h5_path)
    # fp['COCO_val2014_000000014056/fc7'], etc.

    log.infov('Building network...')
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), device_count={'GPU': 1})) as sess:

        # build vgg network
        with tf.name_scope('content_vgg'):
            # TODO remove hardcode and support vgg16
            assert args.network == 'vgg19'
            vgg = vgg19.Vgg19()
            images = tf.placeholder('float', [None, 224, 224, 3])
            vgg.build(images)

        # loop and extract features
        for start in range(0, N, batch_size):
            end = start + batch_size
            if end >= N : end = N
            log.info('%d / %d : %s', start, N, os.path.basename(image_files[start]))

            batch_images = np.zeros((end - start, 224, 224, 3), dtype=np.float32)
            for k in range(start, end):
                im = imread(image_files[k])
                im = imcrop_and_resize(im, 224, 224)
                batch_images[k - start, :, :, :] = im

            conv5_4, fc7, prob = \
                sess.run([vgg.conv5_4, vgg.fc7, vgg.prob], {images: batch_images})

            for k in range(start, end):
                basename = os.path.basename(image_files[k])
                fp.require_group(basename)
                fp[basename]['fc7'] = fc7[k - start, :]
                fp[basename]['conv5_4'] = conv5_4[k - start, :]

            fp.flush()

    fp.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network',
                        choices=['vgg19'], type=str,
                        default='vgg19')
    parser.add_argument('--dataset',
                        choices=['train2014', 'val2014', 'test2015'],
                        default='val2014')
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()

    extract_features(args)


if __name__ == '__main__':
    main()
