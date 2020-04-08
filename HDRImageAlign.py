import glob
import cv2
import rawpy
import numpy as np

# Convert color image into grayscale image using the formula: gray = (19*blue + 183*green + 54*red) / 256
def CvtColorToGray(img_color):
    img_gray = 0.0742*img_color[:, :, 0] + 0.7149*img_color[:, :, 1] + 0.2109*img_color[:, :, 2]
    return img_gray

# Convert image into binary bitmap using the median threshold
def ComputeThresholdBitmap(img_gray):
    # Median threshold
    median = np.median(img_gray)

    # Convert into bitmap according to median threshold
    thres_bitmap = np.array(img_gray)
    thres_bitmap[np.where(thres_bitmap < median)] = 0
    thres_bitmap[np.where(thres_bitmap > median)] = 255

    return thres_bitmap

# Convert image into binary bitmap consists of 0’s wherever the grayscale value is within some specified distance of the threshold, and 1’s elsewhere. 
def ComputeExclusionBitmap(img_gray):
    # Median threshold
    median = np.median(img_gray)

    # Convert into bitmap according to a range around threshold
    ext_bitmap = np.zeros(img_gray.shape)
    img_gray_array = np.array(img_gray)
    ext_bitmap[np.where(img_gray_array < median - 10)] = 255
    ext_bitmap[np.where(img_gray_array > median + 10)] = 255
    
    return ext_bitmap

# Shift a bitmap by (xo,yo) and clear exposed border areas to zero.
def BitmapShift(bitmap, xo, yo):
    translation_matrix = np.float32([[1, 0, xo], [0, 1, yo]])
    shifted_bitmap = cv2.warpAffine(bitmap, translation_matrix, (bitmap.shape[1], bitmap.shape[0]))
    
    return shifted_bitmap

def GetExpShift(img_gray_1, img_gray_2, shift_bits):
    cur_shift = [0, 0]
    shift_ret = [0, 0]

    if shift_bits > 0:
        img_sml_1 = cv2.resize(img_gray_1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        img_sml_2 = cv2.resize(img_gray_2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cur_shift[0], cur_shift[1] = GetExpShift(img_sml_1, img_sml_2, shift_bits-1)
        cur_shift[0] = cur_shift[0] * 2
        cur_shift[1] = cur_shift[1] * 2
    else:
        cur_shift[0] = 0
        cur_shift[1] = 0
        
    thres_bitmap_1 = ComputeThresholdBitmap(img_gray_1)
    ext_bitmap_1 = ComputeExclusionBitmap(img_gray_1)
    thres_bitmap_2 = ComputeThresholdBitmap(img_gray_2)
    ext_bitmap_2 = ComputeExclusionBitmap(img_gray_2)
    
    min_err = img_gray_1.shape[0] * img_gray_1.shape[1]
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            xs = cur_shift[0] + i
            ys = cur_shift[1] + j
            
            shifted_thres_bitmap_2 = BitmapShift(thres_bitmap_2, xs, ys)
            shifted_ext_bitmap_2 = BitmapShift(ext_bitmap_2, xs, ys)
            
            diff_b = cv2.bitwise_xor(thres_bitmap_1, shifted_thres_bitmap_2)
            diff_b = cv2.bitwise_and(diff_b, ext_bitmap_1)
            diff_b = cv2.bitwise_and(diff_b, shifted_ext_bitmap_2)
            
            err = np.count_nonzero(diff_b)
            if err < min_err:
                shift_ret[0] = xs
                shift_ret[1] = ys
                min_err = err
                
    return shift_ret[0], shift_ret[1]

# Align all images in the directory
def AlignImages(dir):
    # Get all the jpg files in the directory
    file_path_dict = {}
    for file_path in glob.glob("./" + dir + "*.CR2"):
        file_name = file_path.split("/")[2].split("\\")[1].split(".")[0]
        curr_exposure = float(file_name.split("_")[2].split("-")[0]) / float(file_name.split("_")[2].split("-")[1])
        file_path_dict.update({curr_exposure : file_path})

    # Sort all the files according to the exposure
    file_path_list = [file_path_dict[exposure] for exposure in sorted(file_path_dict.keys(), reverse=True)]

    # Read the images and convert them into grayscale image
    img_gray_list = []
    for file_path in file_path_list:
        with rawpy.imread(file_path) as raw:
            curr_raw = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            curr_jpg = np.float32(curr_raw / 65535.0*255.0)
            curr_jpg = np.asarray(curr_jpg, np.uint8)
            img_gray_list.append(CvtColorToGray(curr_jpg))

    if len(img_gray_list) <= 1:
        return [0, 0]
        
    # Align all images with the first image
    shift_ret = [0, 0]
    shift_list = []
    shift_list.append([0, 0])
    for i in range(1, len(img_gray_list)):
        shift_ret[0], shift_ret[1] = GetExpShift(img_gray_list[0], img_gray_list[i], 4)
        # translation_matrix = np.float32([[1, 0, shift_ret[0]], [0, 1, shift_ret[1]]])
        # shifted_img = cv2.warpAffine(img_raw_list[i], translation_matrix, (img_raw_list[i].shape[1], img_raw_list[i].shape[0]))
        # shifted_img = np.roll(img_raw_list[i], shift_ret[1], axis=0)
        # shifted_img = np.roll(img_raw_list[i], shift_ret[0], axis=1)
        # if shift_ret[1]>0:
        #     shifted_img[:shift_ret[1], :] = 0
        # elif shift_ret[1]<0:
        #     shifted_img[shift_ret[1]:, :] = 0
        # if shift_ret[0]>0:
        #     shifted_img[:, :shift_ret[0]] = 0
        # elif shift_ret[0]<0:
        #     shifted_img[:, shift_ret[0]:] = 0
        shift_list.append(shift_ret)

    with open("5_shift.txt", "w") as file:
        file.write(str(shift_list))
    
    return shift_list

AlignImages("Images/5_raw/")