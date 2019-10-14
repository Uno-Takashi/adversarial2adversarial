import os
import numpy as np
def clac_perts_slippage(path="precomputing_perturbations/"):
    files = os.listdir(path)
    files_file = [f for f in files if os.path.isfile(os.path.join(path, f))]
    mean_sum=0
    for x in files_file:
        v=np.load(path+x)
        mean_sum+=v.mean()
    print(mean_sum/len(files_file))
    print(len(files_file))

def undo_image_avg(img):
    img_copy = np.copy(img)
    img_copy[:, :, 0] = img_copy[:, :, 0] + 123.68
    img_copy[:, :, 1] = img_copy[:, :, 1] + 116.779
    img_copy[:, :, 2] = img_copy[:, :, 2] + 103.939
    return img_copy

def do_image_avg(img):
    img_copy = np.copy(img)
    img_copy[:, :, 0] = img_copy[:, :, 0] - 123.68
    img_copy[:, :, 1] = img_copy[:, :, 1] - 116.779
    img_copy[:, :, 2] = img_copy[:, :, 2] - 103.939
    img_copy.astype(np.uint8)
    return img_copy

def avg_add_clip_pert(avg_img,v):
    clipped_v = np.clip(undo_image_avg(avg_img[0,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(avg_img[0,:,:,:]), 0, 255)
    pert_img =  avg_img+ clipped_v[None, :, :, :]
    return pert_img[0]

def get_random_pert(path="precomputing_perturbations/"):
    target=np.random.randint(1000)+1
    v=np.load(path+"universal-target-"+str(target).zfill(5)+".npy")
    return v
