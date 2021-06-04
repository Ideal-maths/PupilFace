import cv2
import numpy as np
import torch


def letterbox_image(image, size):
    ih, iw, _ = np.shape(image)
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones([size[1],size[0],3])*128
    new_image[(h-nh)//2:nh+(h-nh)//2, (w-nw)//2:nw+(w-nw)//2] = image
    return new_image
    
def retinaface_correct_boxes(result, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    
    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0]]

    offset_for_boxs = [offset[1], offset[0], offset[1],offset[0]]
    offset_for_landmarks = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0]]

    result[:,:4] = (result[:,:4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:,5:] = (result[:,5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result
    
def point_form(boxes):
    #------------------------------#
    #   Get the upper left and lower right corners of the box
    #------------------------------#
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,
                     boxes[:, :2] + boxes[:, 2:]/2), 1)


def center_size(boxes):
    #------------------------------#
    #   Get the center and width and height of the box
    #------------------------------#
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,
                     boxes[:, 2:] - boxes[:, :2], 1)


def intersect(box_a, box_b):
    # Calculates the intersection area of all real boxes and priori boxes
    A = box_a.size(0)
    B = box_b.size(0)
    #------------------------------#
    #   Gets the upper left corner of the intersecting rectangle
    #------------------------------#
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    #------------------------------#
    #   Gets the bottom right corner of the intersecting rectangle
    #------------------------------#
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    #-------------------------------------#
    #   Calculate the coincidence area between the priori box and all the real boxes
    #-------------------------------------#
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):

    inter = intersect(box_a, box_b)
    #-------------------------------------#
    #   Calculate the respective area of the anchor and the bounding box
    #-------------------------------------#
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    union = area_a + area_b - inter
    #-------------------------------------#
    #   The intersection and union ratio of each bounding box to A anchor is [A,B]
    #-------------------------------------#
    return inter / union  # [A,B]

def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    #----------------------------------------------#
    #   Calculate the degree of coincidence between the anchor and the bounding box
    #----------------------------------------------#
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    #----------------------------------------------#
    #   The best degree of overlap between all bounding boxes and anchor
    #   best_prior_overlap [truth_box,1]
    #   best_prior_idx [truth_box,1]
    #----------------------------------------------#
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)


    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]            
    # Shape: [num_priors] 此处为每一个anchor对应的label取出来
    conf = labels[best_truth_idx]        
    matches_landm = landms[best_truth_idx]

    conf[best_truth_overlap < threshold] = 0    

    loc = encode(matches, priors, variances)
    landm = encode_landm(matches_landm, priors, variances)

    # [num_priors, 4]
    loc_t[idx] = loc
    # [num_priors]
    conf_t[idx] = conf
    # [num_priors, 10]
    landm_t[idx] = landm


def encode(matched, priors, variances):

    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]

    g_cxcy /= (variances[0] * priors[:, 2:])
    

    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def encode_landm(matched, priors, variances):

    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)


    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    return g_cxcy

def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

def decode(loc, priors, variances):

    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                    priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):

    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def non_max_suppression(boxes, conf_thres=0.5, nms_thres=0.3):
    detection = boxes

    mask = detection[:, 4] >= conf_thres
    detection = detection[mask]
    if not np.shape(detection)[0]:
        return []

    best_box = []
    scores = detection[:, 4]

    arg_sort = np.argsort(scores)[::-1]
    detection = detection[arg_sort]

    while np.shape(detection)[0]>0:

        best_box.append(detection[0])
        if len(detection) == 1:
            break
        ious = iou(best_box[-1], detection[1:])
        detection = detection[1:][ious<nms_thres]

    return np.array(best_box)

def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    
    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
    
    iou = inter_area/np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou
