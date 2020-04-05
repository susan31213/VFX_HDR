import glob
import cv2
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
    for file_path in glob.glob("./" + dir + "*.JPG"):
        file_name = file_path.split("/")[2].split("\\")[1].split(".")[0]
        curr_exposure = float(file_name.split("_")[2].split("-")[0]) / float(file_name.split("_")[2].split("-")[1])
        file_path_dict.update({curr_exposure : file_path})

    # Sort all the files according to the exposure
    file_path_list = [file_path_dict[exposure] for exposure in sorted(file_path_dict.keys(), reverse=True)]

    # Read the images and convert them into grayscale image
    img_color_list = []
    img_gray_list = []
    for file_path in file_path_list:
        curr_img = cv2.imread(file_path)
        img_color_list.append(curr_img)
        img_gray_list.append(CvtColorToGray(curr_img))

    if len(img_color_list) <= 1:
        return img_color_list
        
    # Align all images with the first image
    shift_ret = [0, 0]
    img_shifted_list = []
    img_shifted_list.append(img_color_list[0])
    # cv2.imwrite("Images/8_aligned/shifted_" + file_path_list[0].split("/")[2].split("\\")[1], img_color_list[0])
    for i in range(1, len(img_color_list)):
        shift_ret[0], shift_ret[1] = GetExpShift(img_gray_list[0], img_gray_list[i], 4)
        translation_matrix = np.float32([[1, 0, shift_ret[0]], [0, 1, shift_ret[1]]])
        shifted_img = cv2.warpAffine(img_color_list[i], translation_matrix, (img_color_list[i].shape[1], img_color_list[i].shape[0]))
        img_shifted_list.append(shifted_img)
        # cv2.imwrite("Images/8_aligned/shifted_" + file_path_list[i].split("/")[2].split("\\")[1], shifted_img)
    
    return img_shifted_list

# my_list = AlignImages("Images/8/")