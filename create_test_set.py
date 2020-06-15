from argparse import ArgumentParser
import cv2
import numpy as np
import os

parser = ArgumentParser()
parser.add_argument('--input_dir', type=str, help='Directory where HR images are kept.')
parser.add_argument('--output_dir', type=str, help='Directory where to output downsampled images.')
parser.add_argument('--scale_factor', type=int, help='Choose downsampling factor.')

def main():
    args = parser.parse_args()
    image_paths = [os.path.join(args.input_dir,x) for x in os.listdir(args.input_dir)]

    for image in image_paths:
        high_res = cv2.imread(image, 1)
        width = int(high_res.shape[1]//args.scale_factor)
        height = int(high_res.shape[0])
        dim = (width,height)
        low_res = cv2.resize(high_res, dim, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(image)), low_res)

if __name__ == '__main__':
    main()
