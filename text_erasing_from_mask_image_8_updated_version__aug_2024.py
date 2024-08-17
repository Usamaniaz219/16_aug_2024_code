import numpy as np
import cv2
import os
import time
import EasyOcr_Bounding_box_into_mask
from EasyOcr_Bounding_box_into_mask import main


# def text_eraser_from_mask_images(source_image,mask_image,output_image):
    
#     image_Gray = cv2.cvtColor(source_image,cv2.COLOR_BGR2GRAY)
#     height, width = image_Gray.shape[:2]
#     # Create a blank (black) image
#     blank_image = np.zeros((height, width), dtype=np.uint8)


#     image_Gray = cv2.cvtColor(source_image,cv2.COLOR_BGR2GRAY)
#     # inverted_image_Gray = cv2.bitwise_not(image_Gray)
#     height, width = image_Gray.shape[:2]
#     kernel = np.ones((3, 3), np.uint8)

#     thresh1 = cv2.adaptiveThreshold(image_Gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

#     output = cv2.bitwise_and(thresh1,output_image)
#     # output = cv2.dilate(output,kernel,iterations=2)
#     _,output = cv2.threshold(output,10,255,cv2.THRESH_BINARY)
#     # cv2.imwrite(f"output_mask.jpg",output)
#     contours,hierarchy = cv2.findContours(output,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#     print("Original contours length",len(contours))
#     retained_contours = []
#     for i,cont in enumerate(contours):
   
#         image_with_text = blank_image.copy()
#         cnt = np.array([cont], np.int32)
#         cnt = cnt.reshape((-1, 1, 2))

#         cv2.fillPoly(image_with_text, [cnt], (255,255,255))

#         # filled_area = cv2.countNonZero(image_with_text)
#         # # total image area
#         # total_image_area = image_Gray.shape[0]*image_Gray.shape[1]
#         # if filled_area <= 0.1*total_image_area:

#         result_image = cv2.bitwise_and(image_with_text, mask_image) # Perform logical AND operation with the source mask image
        
#         if np.any(result_image):  # Check if the result image is blank
#             retained_contours.append(cont)
#     return retained_contours



# def retain_intersected_contours(retained_contours,source_mask_image):
#     height,width = source_mask_image.shape[:2]
#     # print("Source Mask Image shape",source_mask_image)
#     blank_image = np.zeros((height,width),dtype=np.uint8)
        
    
#     for i,cnt211 in enumerate(retained_contours):
#         # print(cnt211)
#         blank_image_with_text = blank_image.copy()
#         cnt11 = np.array([cnt211], np.int32)
#         cnt11 = cnt11.reshape((-1, 1, 2))

#         cv2.fillPoly(blank_image_with_text, [cnt11], (255))
    
#         intersections = cv2.bitwise_and(blank_image_with_text,source_mask_image)
#         intersection_area = np.sum(intersections)
#         text_mask_area = np.sum(blank_image_with_text)
#         # print("bbox_mask_intersection_area",bbox_mask_intersection_area)
#         if text_mask_area==0:
#             return 0
#         intersection_percentage = intersection_area/text_mask_area
#         # print("intersection percentage",intersection_percentage)
#         if intersection_percentage >=0.5:
#             intersected_contours.append(cnt211)
#             # print("appended")

#     return intersected_contours

retain_contours1 = []


def retain_contours(source_image,mask_image,bounding_boxes):
    image_Gray = cv2.cvtColor(source_image,cv2.COLOR_BGR2GRAY)
    source_image_copy = source_image.copy()
    height, width = image_Gray.shape[:2]
    # Create a blank (black) image
    blank_image = np.zeros((height, width), dtype=np.uint8)
    thresh1 = cv2.adaptiveThreshold(image_Gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    for box in bounding_boxes:
        blank_image_bbox = blank_image.copy()
        blank_image_copy = blank_image.copy()
        box = np.array(box, dtype=np.int32)
        box = box.reshape((-1, 1, 2))
        cv2.fillPoly(blank_image_bbox, [box], 255)
        intersections_with_bbox = cv2.bitwise_and(blank_image_bbox,mask_image)
        intersection_area_bbox = np.sum(intersections_with_bbox)
        bbox_mask_area = np.sum(blank_image_bbox)
        # print("bbox_mask_intersection_area",bbox_mask_area)
        if bbox_mask_area==0:
            return 0
        intersection_percentage_bbox = intersection_area_bbox/bbox_mask_area
        # print("intersection percentage",intersection_percentage_bbox)
        if intersection_percentage_bbox >=0.5:
            cv2.polylines(source_image_copy, [box], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.imwrite("source_image.jpg",source_image_copy)
            
            cv2.fillPoly(blank_image_copy, [box], 255)
            cv2.imwrite("blank_image.jpg",blank_image_bbox)
            output = cv2.bitwise_and(thresh1,blank_image_copy)
            cv2.imwrite("merged_image.jpg",output)

            _,output = cv2.threshold(output,10,255,cv2.THRESH_BINARY)

            contours,hierarchy = cv2.findContours(output,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            # print("Original contours length",len(contours))
            # retained_contours = []
            for i,cont in enumerate(contours):
        
                image_with_text = blank_image.copy()
                cnt = np.array([cont], np.int32)
                cnt = cnt.reshape((-1, 1, 2))

                cv2.fillPoly(image_with_text, [cnt], (255,255,255))

                # filled_area = cv2.countNonZero(image_with_text)
                # # total image area
                # total_image_area = image_Gray.shape[0]*image_Gray.shape[1]
                # if filled_area <= 0.1*total_image_area:

                result_image = cv2.bitwise_and(image_with_text, mask_image) # Perform logical AND operation with the source mask image
                
                if np.any(result_image):  # Check if the result image is blank
                    retain_contours1.append(cont)
            return retain_contours1
        
intersected_contours = []
        

def retain_intersected_contours(retained_contours,source_mask_image):
    height,width = source_mask_image.shape[:2]
    # print("Source Mask Image shape",source_mask_image)
    blank_image = np.zeros((height,width),dtype=np.uint8)
    if retained_contours is not None:
    
        for i,cnt211 in enumerate(retained_contours):
            # print(cnt211)
            blank_image_with_text = blank_image.copy()
            cnt11 = np.array([cnt211], np.int32)
            cnt11 = cnt11.reshape((-1, 1, 2))

            cv2.fillPoly(blank_image_with_text, [cnt11], (255))
        
            intersections = cv2.bitwise_and(blank_image_with_text,source_mask_image)
            cv2.imwrite("intersected_image.jpg",intersections)

            intersection_area = np.sum(intersections)
            text_mask_area = np.sum(blank_image_with_text)
            # print("bbox_mask_intersection_area",bbox_mask_intersection_area)
            if text_mask_area==0:
                return 0
            intersection_percentage = intersection_area/text_mask_area
            # print("intersection percentage",intersection_percentage)
            if intersection_percentage >=0.5:
                intersected_contours.append(cnt211)
                # print("appended")

        return intersected_contours






















#########################################################################################################################################

# def retain_intersected_contours(retained_contours,source_mask_image,bounding_boxes,source_image):
#     image_Gray = cv2.cvtColor(source_image,cv2.COLOR_BGR2GRAY)
#     height,width = source_mask_image.shape[:2]
#     source_image_copy = source_image
#     # source_image_copy_gray = cv2.cvtColor(source_image_copy,cv2.COLOR_BGR2GRAY)
#     # print("Source Mask Image shape",source_mask_image)
#     blank_image = np.zeros((height,width),dtype=np.uint8)
#     for box in bounding_boxes:
#         blank_image_bbox = blank_image.copy()
#         box = np.array(box, dtype=np.int32)
#         box = box.reshape((-1, 1, 2))

# # Draw the contours on the mask
#         # cv2.drawContours(mask, [box], contourIdx=-1, color=255, thickness=2)
#         # cv2.polylines(blank_image_bbox, [box], isClosed=True, color=(255, 255, 255), thickness=2)
#         cv2.fillPoly(blank_image_bbox, [box], 255)
#         intersections_with_bbox = cv2.bitwise_and(blank_image_bbox,source_mask_image)
#         intersection_area_bbox = np.sum(intersections_with_bbox)
#         bbox_mask_area = np.sum(blank_image_bbox)
#         # print("bbox_mask_intersection_area",bbox_mask_area)
#         if bbox_mask_area==0:
#             return 0
#         intersection_percentage_bbox = intersection_area_bbox/bbox_mask_area
#         print("intersection percentage",intersection_percentage_bbox)
#         if intersection_percentage_bbox >=0.5:
#             cv2.polylines(source_image_copy, [box], isClosed=True, color=(255, 0, 0), thickness=2)
#             cv2.imwrite("blank_image.jpg",source_image_copy)
#             for i,cnt211 in enumerate(retained_contours):
#                 # print(cnt211)
#                 blank_image_with_text = blank_image.copy()
#                 cnt11 = np.array([cnt211], np.int32)
#                 cnt11 = cnt11.reshape((-1, 1, 2))

#                 cv2.fillPoly(blank_image_with_text, [cnt11], (255))
            
#                 intersections_with_text = cv2.bitwise_and(blank_image_with_text,source_mask_image)
#                 intersection_area_text = np.sum(intersections_with_text)
#                 text_mask_area = np.sum(blank_image_with_text)
#                 # print("bbox_mask_intersection_area",bbox_mask_intersection_area)
#                 if text_mask_area==0:
#                     return 0
#                 intersection_percentage = intersection_area_text/text_mask_area
#                 # print("intersection percentage",intersection_percentage)
#                 if intersection_percentage >=0.5:
#                     intersected_contours.append(cnt211)
#     return intersected_contours


#########################################################################################################################3


# print("length of intersected Contours",len(intersected_contours))

def draw_intersected_contours_on_mask_image(mask_image,intersected_contours):
    # mask_image = cv2.medianBlur(mask_image,5)

    height,width = mask_image.shape[:2]
    blank_mask_image = np.zeros((height,width),dtype = np.uint8)
    count = 0

    if intersected_contours is not None:
    
        
        for i,cnt1 in enumerate(intersected_contours):
            cnt11 = np.array([cnt1], np.int32)
            cnt11 = cnt11.reshape((-1, 1, 2))

            cv2.fillPoly(blank_mask_image, [cnt11], (255))
            # cv2.dilate(blank_mask_image,kernel,iterations=1)
            cv2.fillPoly(mask_image, [cnt11], (255))
        
        # kernel = np.ones((3, 3), np.uint8)
        count+=1

        cv2.imwrite(f"text_intersected_area_masks_ca_colma_{count}.jpg",blank_mask_image)
        # mask_image_dilated = cv2.dilate(blank_mask_image,kernel,iterations=2)
        # merged_result = cv2.bitwise_or(mask_image_dilated,mask_image)
    
        # return merged_result

    return mask_image

def process_image(source_image_path, source_mask_path, output_dir):
    ori_image_name = os.path.splitext(os.path.basename(source_image_path))[0]
    print(f"Processing ori image name image: {ori_image_name}")

    source_image = cv2.imread(source_image_path)
    
    # cv2.imwrite("mask-temp.jpg",source_image)
    if source_image is None:
        print(f"Error reading mask image: {source_image_path}")
        return None
    
 
    mask_image = cv2.imread(source_mask_path)

    if mask_image is None:
        print(f"Error reading bounding box image: {source_mask_path}")
        return None
    mask_image = cv2.cvtColor(mask_image,cv2.COLOR_BGR2GRAY)
    bounding_boxes, output_image = main(source_image, tile_width, tile_height)
    retained_contours = retain_contours(source_image,mask_image,bounding_boxes)
    # intersected_contours = retain_intersected_contours(retained_contours,mask_image)
    intersected_contours = retain_intersected_contours(retained_contours,mask_image)
    # if intersected_contours is not None:
    #     print("intersected_retained_contours",len(intersected_contours))
    # print("Length of retained Contours",len(retained_contours))
    merged_result = draw_intersected_contours_on_mask_image(mask_image,intersected_contours)
    if intersected_contours is not None:
        intersected_contours.clear()
    if retained_contours is not None: 
        retained_contours.clear()
    return merged_result



def process_images(input_dir, output_dir, bounding_box_dir):
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    file_count = 0
    
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # original_file_list.append(filename)
                file_count += 1
                image_path = os.path.join(input_dir, filename)
                ori_image_name = os.path.splitext(os.path.basename(image_path))[0]
                print("mask image name :",ori_image_name)
            
                for root,dirs, files in os.walk(bounding_box_dir):              
                    # all_masks = os.listdir(bounding_box_dir)
                    for dir in dirs:
                        if dir==ori_image_name:
                            dir1 = os.path.join(root,dir)
                            # mask_dirs.append(dir1)
                            
                            all_masks = os.listdir(dir1)
                            masks_renamed = [mask.replace(".jpg","").replace(".png","") for mask in all_masks]

                            for renamed_mask in masks_renamed:
                                mask_path = f"{bounding_box_dir}/{dir}/{renamed_mask}.jpg"
                                output_ = process_image(image_path, mask_path, output_dir)
                                if output_ is None:
                                    continue

                                # output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
                                output_subdir = os.path.join(output_dir, ori_image_name)
                                # output_subdir = f"{output_subdir}_{renamed_mask}_intersection_of_0.1"
                                os.makedirs(output_subdir, exist_ok=True)
                                
                                output_file_path = os.path.join(output_subdir, f"{renamed_mask}_output_mask.jpg")
                                cv2.imwrite(output_file_path,output_)
                                
                                print(f"Processed {filename} in {time.time() - start_time:.2f} seconds")
                        continue
                    break    



if __name__ == "__main__":
    tile_width = 1024
    tile_height = 1024

    input__image_directory = '/home/usama/Converted_1_jpg_from_tiff_july3_2024_updated'
    mask_image_dir = '/home/usama/9_aug_2024/'
    output_directory = '/media/usama/6EDEC3CBDEC389B3/16_aug_2024_results_11/'

    # input__image_directory = '/home/usama/Data_13_aug_2024_temp/A/'
    # mask_image_dir = '/home/usama/Data_13_aug_2024_temp/B/'
    # output_directory = '/media/usama/6EDEC3CBDEC389B3/13_aug_2024_results_11/'
    
    process_images(input__image_directory, output_directory, mask_image_dir)


