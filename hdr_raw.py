import rawpy
import glob
import cv2
import numpy as np 
from tqdm import tqdm

dir = "Images/1_raw/"

# Get all the jpg files in the directory
file_path_dict = {}
for file_path in glob.glob("./" + dir + "*.CR2"):
    file_name = file_path.split("/")[2].split("\\")[1].split(".")[0]
    curr_exposure = float(file_name.split("_")[2].split("-")[0]) / float(file_name.split("_")[2].split("-")[1])
    file_path_dict.update({curr_exposure : file_path})

# Sort all the files according to the exposure
exposure_list = sorted(file_path_dict.keys(), reverse=True)
file_path_list = [file_path_dict[exposure] for exposure in exposure_list]

# Read the images and calculate x
img_raw_list = []
img_x_list = []
for i in range(len(file_path_list)):
    file_path = file_path_list[i]
    with rawpy.imread(file_path) as raw:
        curr_raw = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        curr_raw = curr_raw.astype('float32') / 2**16
        img_raw_list.append(curr_raw)
        curr_x = np.array([raw_value / exposure_list[i] for raw_value in curr_raw])
        img_x_list.append(curr_x)

# Calculate radiance map
img_num = len(img_raw_list)
img_shape = img_raw_list[0].shape
radiance_map = np.zeros([img_shape[0], img_shape[1]], dtype=np.float64)
img_hdr = np.zeros(img_shape, dtype=np.float64)

t_square = np.array([exposure_list[i] ** 2 for i in range(img_num)])
sum_t_square = np.sum(t_square)

for channel in range(img_shape[2]):
    for i in tqdm(range(img_shape[0])):
        for j in range(img_shape[1]):
            tx = np.array([exposure_list[k] * img_x_list[k][i, j, channel] for k in range(img_num)])
            sum_tx = np.sum(tx)
            radiance_map[i, j] = sum_tx / sum_t_square
    img_hdr[..., channel] = cv2.normalize(radiance_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

cv2.imwrite("raw_hdr.hdr", img_hdr[...,::-1])

# hdr_image = cv2.imread("raw_hdr.hdr")
cv2.imwrite("hdr_jpg.jpg", img_hdr[...,::-1].astype(np.uint8))

# np.save("radiance_map_1", radiance_map)
# print(max_sum_tx)
# radiance_map = np.load("radiance_map_1.npy")
# print(np.amax(radiance_map))
# print(np.amax(img_raw_list[0]))

# radiance_map[np.where(radiance_map > 65535)] = 65535

# raw.raw_image[:,:] = img_raw_list[0]
# im = raw.postprocess(use_camera_wb=True, no_auto_bright=True)

# img = np.fromfile(radiance_map, np.dtype('u1'), img_raw_list[0].shape)
# colour = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
# cv2.imwrite("radiance.jpg", colour)