import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import face_alignment


class FaceMarker(object):

    def __init__(self):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=False)

    def mark(self, inp_img):
        x = cv2.resize(inp_img, (256,256))
        preds = self.fa.get_landmarks(x)
        
        if preds is not None:
            preds = preds[0]
            #mask = np.zeros_like(x)
            mask = np.full_like(x,40)
            
            # Draw right eye mask
            pnts_right = [(preds[i,0],preds[i,1]) for i in range(36,42)]
            hull = cv2.convexHull(np.array(pnts_right)).astype(np.int32)
            mask = cv2.drawContours(mask,[hull],0,(255,0,0),-1)

            # Draw left eye mask
            pnts_left = [(preds[i,0],preds[i,1]) for i in range(42,48)]
            hull = cv2.convexHull(np.array(pnts_left)).astype(np.int32)
            mask = cv2.drawContours(mask,[hull],0,(255,0,0),-1)

            # Draw mouth mask
            pnts_mouth = [(preds[i,0],preds[i,1]) for i in range(48,60)]
            hull = cv2.convexHull(np.array(pnts_mouth)).astype(np.int32)
            mask = cv2.drawContours(mask,[hull],0,(0,255,0),-1)

            # Draw nose mask
            pnts_nose = [(preds[i,0],preds[i,1]) for i in range(27,36)]
            hull = cv2.convexHull(np.array(pnts_nose)).astype(np.int32)
            mask = cv2.drawContours(mask,[hull],0,(0,0,255),-1)

            mask = cv2.dilate(mask, np.ones((14,14), np.uint8), iterations=1)
            mask = cv2.GaussianBlur(mask, (7,7), 0)
            
        else:
            mask = np.zeros_like(x)

        return mask