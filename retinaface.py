import colorsys
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from nets.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.box_utils import (decode, decode_landm, letterbox_image,
                             non_max_suppression, retinaface_correct_boxes)
from utils.config import cfg_mnet, cfg_re50


def preprocess_input(image):
    image -= np.array((104, 117, 123),np.float32)
    return image


class Retinaface(object):
    _defaults = {
        "model_path"        : 'logs/Epoch113-Total_Loss5.4983.pth',
        "backbone"          : 'mobilenet',
        "confidence"        : 0.5,
        "nms_iou"           : 0.45,
        "cuda"              : True,

        "input_shape"       : [1280, 1280, 3],
        "letterbox_image"   : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50
        self.generate()
        if self.letterbox_image:
            self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()

    #---------------------------------------------------#
    #   load the model
    #---------------------------------------------------#
    def generate(self):
        self.net = RetinaFace(cfg=self.cfg, mode='eval').eval()

        #-------------------------------#
        #   load model and weight
        #-------------------------------#
        print('Loading weights into state dict...')
        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict)
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0，1'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        print('Finished!')

    #---------------------------------------------------#
    #   detect images
    #---------------------------------------------------#
    def detect_image(self, image):

        old_image = image.copy()

        image = np.array(image,np.float32)

        #---------------------------------------------------#
        #   calculate scale to predect the orginal picture weight and height
        #---------------------------------------------------#
        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]]

        im_height, im_width, _ = np.shape(image)


        if self.letterbox_image:
            image = np.array(letterbox_image(image, [self.input_shape[1], self.input_shape[0]]), np.float32)
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        with torch.no_grad():
            #-----------------------------------------------------------#
            #   normalization
            #-----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0)

            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()

            loc, conf, landms = self.net(image)
            
            #-----------------------------------------------------------#
            #   decode for predicted result
            #-----------------------------------------------------------#
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes = boxes.cpu().numpy()

            conf = conf.data.squeeze(0)[:,1:2].cpu().numpy()
            
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            landms = landms.cpu().numpy()

            boxes_conf_landms = np.concatenate([boxes, conf, landms],-1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            if len(boxes_conf_landms)<=0:
                return old_image

            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
            
        boxes_conf_landms[:,:4] = boxes_conf_landms[:,:4]*scale
        boxes_conf_landms[:,5:] = boxes_conf_landms[:,5:]*scale_for_landmarks

        for b in boxes_conf_landms:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))

            # b[0]-b[3] are the cordinate of facial box,b[4] is the score
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            print(b[0], b[1], b[2], b[3], b[4])
            # b[5]-b[14] are the cordinate of facial landmarks
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
        return old_image
