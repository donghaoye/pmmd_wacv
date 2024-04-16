


import cv2
import os
import numpy as np


# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

if __name__ == '__main__':

    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (600, 100)
    fontScale              = 3
    fontColor              = (255, 0, 0)
    thickness              = 5
    lineType               = 3

    # source_img_root = "/data/datasets/AGORA/images/validation"
    
    proj_root = "/home/haoye/codes/ads/"
    gt_result_root = f"{proj_root}/output/front_com/pred_mesh_gt"
    crmh_result_root = f"{proj_root}/output/front_com/pred_mesh_crmh"
    romp_result_root = f"{proj_root}/output/front_com/pred_mesh_romp"
    bev_result_root = f"{proj_root}/output/front_com/pred_mesh_bev"
    # ours_result_root = f"{proj_root}/output/front_com/pred_mesh_ours_0.5"
    ours_result_root = "./figs/fig5_frontview/ours"

    '''
    /data/datasets/AGORA/images/validation/ag_validationset_renderpeople_bfh_archviz_5_10_cam02_00000_1280x720.png
    BEV  /data/datasets/AGORA/BEV_testset_outputs/ag_validationset_renderpeople_bfh_archviz_5_10_cam02_00000_1280x720__2_0.08.png
    ROMP /data/datasets/AGORA/ROMP_testset_outputs/ag_validationset_renderpeople_bfh_archviz_5_10_cam02_00000_1280x720.png
    CRMH /data/datasets/AGORA/multihuman_testset_outputs/ag_validationset_renderpeople_bfh_archviz_5_10_cam02_00000_1280x720.png.output.jpg
    Ours /data/datasets/AGORA/ADRT_w_mask_conf_outputs_0.5/ag_validationset_renderpeople_bfh_archviz_5_10_cam02_00000_1280x720.output.png
         /data/datasets/AGORA/ADRT_w_mask_conf_outputs_0.5/ag_validationset_renderpeople_bfh_archviz_5_10_cam02_00001_1280x720.output.png
    '''
    output_fold = "./figs/fig5_frontview/front_view_compare_src_crmh_romp_bev_ours"
    os.makedirs(output_fold, exist_ok=True)

    img_name_list = os.listdir(bev_result_root)
    img_name_list = sorted(img_name_list)

    for img_name in img_name_list:
        
        gt_img_path = os.path.join(gt_result_root, img_name[:-4] + '.output.png')
        crmh_img_path = os.path.join(crmh_result_root, img_name)
        romp_img_path = os.path.join(romp_result_root, img_name)
        bev_img_path = os.path.join(bev_result_root, img_name)
        ours_img_path = os.path.join(ours_result_root, img_name + '.output.png')
        
        gt_img = cv2.imread(gt_img_path)
        crmh_img = cv2.imread(crmh_img_path)
        romp_img = cv2.imread(romp_img_path)
        bev_img = cv2.imread(bev_img_path)
        ours_img = cv2.imread(ours_img_path)
        if bev_img is None or romp_img is None or ours_img is None:
            print(ours_img_path, "not exists...")
            continue

        print(bev_img.shape)
        # gt_img = gt_img[16:-(22+6), :832*2, :]
        crmh_img = crmh_img[22:-22, :832*2, :]
        # crmh_img = crmh_img[22:-22, 832:832*2, :]
        romp_img = romp_img[22:-22, 832:832*2, :]
        bev_img = bev_img[22:-22, 832:832*2, :]
        ours_img = ours_img[16:-(22+6), 832:832*2, :]
        

        # all_imgs = cv2.hconcat([gt_img, crmh_img, romp_img, bev_img, ours_img])
        all_imgs = cv2.hconcat([crmh_img, romp_img, bev_img, ours_img])
        try:
            output_path = os.path.join(output_fold, img_name[:-4] + "src_gt_crmh_romp_bev_ours.png")
            cv2.imwrite(output_path, all_imgs)
            print(output_path)
            print("-" * 50)
        except:
            print("exp....")
       



        # print(src_img_path)
        # print(bev_img_path)
        # print(romp_img_path)
        # print(crmh_img_path)
        # print("src_img.shape: ", src_img.shape)
        # print("bev_img.shape: ", bev_img.shape)
        # print("romp_img.shape: ", romp_img.shape)
        # print("crmh_img.shape: ", crmh_img.shape)
        # print("crmh_img_pad.shape: ", crmh_img_pad.shape)
        # src_img.shape:  (720, 1280, 3)
        # bev_img.shape:  (720, 3280, 3)
        # romp_img.shape:  (720, 2560, 3)
        # crmh_img.shape:  (1536, 832, 3)



