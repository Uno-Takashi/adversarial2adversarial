import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
from noise_model import get_noise_model

from perts_util import *
from PIL import Image
from matplotlib import pylab as plt


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, default="E:/imagenet/tst/",
                        help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, default="checkpoints/weights.022-396886.438-6.27580.hdf5",
                        help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="advx,0,1",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args()
    return args

def img_resize(np_img):
    pil_img=Image.fromarray(np_img)
    img_resize = pil_img.resize((224, 224))
    return np.asarray(img_resize)
def advx_noise(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    advx_slippage=0.04141668473257668
    img=img_resize (img)
    img = img.astype(np.float)
    avg_img=do_image_avg(img)
    pert_noise=get_random_pert()
    pert_noise+=advx_slippage
    pert_noise=pert_noise*np.random.beta(3,1)
    noise_img=avg_add_clip_pert(avg_img.reshape(1,224,224,3),pert_noise).astype(np.uint8)
    noise_img=undo_image_avg(noise_img).astype(dtype='uint8')
    noise_img = cv2.cvtColor(noise_img, cv2.COLOR_RGB2BGR)
    #noise_img=avg_img+get_random_pert()[0]
    #noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    return noise_img

def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main():
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file
    val_noise_model = get_noise_model(args.test_noise_model)
    model = get_model(args.model)
    model.load_weights(weight_file)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        h, w, _ = image.shape
        image = cv2.resize(image,(224,224))
        h, w, _ = image.shape


        out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
        noise_image = val_noise_model(image)
        pred = model.predict(np.expand_dims(noise_image, 0))
        denoised_image = get_image(pred[0])
        out_image[:, :w] = image
        out_image[:, w:w * 2] = noise_image
        out_image[:, w * 2:] = denoised_image

        if args.output_dir:
            cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", out_image)
        else:
            cv2.imshow("result", out_image)
            key = cv2.waitKey(-1)
            # "q": quit
            if key == 113:
                return 0


if __name__ == '__main__':
    main()
