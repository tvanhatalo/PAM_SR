from argparse import ArgumentParser
import cv2
import numpy as np
import os

# code to convert HR dataset to grayscale

parser = ArgumentParser()
parser.add_argument('--rgb_dir', type=str, help='Directory where HR RGB images are kept.')
parser.add_argument('--gray_dir', type=str, help='Directory for grascale images.')


def main():
    args = parser.parse_args()
    image_paths = [os.path.join(args.rgb_dir,x) for x in os.listdir(args.rgb_dir)]

    for image in image_paths:
        rgb_img = cv2.imread(image,1)
        print(rgb_img.shape)
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(args.gray_dir, os.path.basename(image)), gray_img)
        print(gray_img.shape)

if __name__ == '__main__':
    main()
        
