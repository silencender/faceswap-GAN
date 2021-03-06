from color_correction import *
import cv2
import numpy as np
from vc_utils import cal_roi
import matplotlib.pyplot as plt

class FaceTransformer(object):
    """
    Attributes:
        path_func: string, direction for the transformation: either AtoB or BtoA.
        model: the generator of the faceswap-GAN model
    """
    def __init__(self): 
        self.path_func = None
        self.model = None
        
        self.inp_img = None
        self.input_size = None
        self.img_bgr = None
        self.roi = None
        self.roi_size = None
        self.ae_input = None
        self.ae_output = None
        self.ae_output_masked = None
        self.ae_output_bgr = None
        self.ae_output_a = None
        self.result = None 
        self.result_rawRGB = None
        self.result_alpha = None
    
    def set_model(self, model):
        self.model = model

    def _preprocess_inp_img(self, inp_img, inp_layout, roi_coverage, IMAGE_SHAPE):
        img_bgr = cv2.cvtColor(inp_img, cv2.COLOR_RGB2BGR)
        layout_bgr = cv2.cvtColor(inp_layout, cv2.COLOR_RGB2BGR)
        #layout_bgr = inp_layout
        input_size = img_bgr.shape
        input_layout_size = layout_bgr.shape
        roi_x0, roi_x1, roi_y0, roi_y1 = cal_roi(input_size, roi_coverage)
        roi_x0_l, roi_x1_l, roi_y0_l, roi_y1_l = cal_roi(input_layout_size, roi_coverage)
        roi = img_bgr[roi_x0:roi_x1, roi_y0:roi_y1,:] # BGR, [0, 255]  
        roi_layout = layout_bgr[roi_x0_l:roi_x1_l, roi_y0_l:roi_y1_l,:]
        roi_size = roi.shape
        roi_bound = (roi_x0, roi_x1, roi_y0, roi_y1)
        ae_input = cv2.resize(roi, IMAGE_SHAPE[:2])/255. * 2 - 1 # BGR, [-1, 1]  
        ae_layout_input = cv2.resize(roi_layout, IMAGE_SHAPE[:2])/255.

        # = cv2.cvtColor(((ae_input+1)/2.*255).astype('uint8'), cv2.COLOR_BGR2RGB)
        #y = cv2.cvtColor(((ae_layout_input+1)/2.*255).astype('uint8'), cv2.COLOR_BGR2RGB)
        #c = np.clip((0.8*y+x),0,255) .astype('uint8')
        #p=np.hstack([x,y,c])
        #plt.imshow(p)
        #exit()

        #ae_input = cv2.resize(img_bgr, IMAGE_SHAPE[:2])/255. * 2 - 1 # BGR, [-1, 1]  
        self.img_bgr = img_bgr
        self.input_size = input_size
        self.roi = roi
        self.roi_size = roi_size
        self.roi_bound = roi_bound
        self.ae_input = ae_input
        self.ae_layout_input = ae_layout_input
    
    def _ae_forward_pass(self, ae_input, ae_layout_input):
        ae_out = self.path_func([[ae_input], [ae_layout_input]])
        ae_out = np.squeeze(np.array([ae_out]))
        #ae_out = cv2.resize(ae_out, self.input_size[:2])
        #roi_x0, roi_x1, roi_y0, roi_y1 = self.roi_bound
        self.ae_output = ae_out #[roi_x0:roi_x1, roi_y0:roi_y1,:]
        
    def _postprocess_roi_img(self, ae_output, roi, roi_size, color_correction):
        ae_output_a = ae_output[:,:,0] * 255
        #ae_output_a = ae_output_a[...,np.newaxis]
        ae_output_a = cv2.resize(ae_output_a, (roi_size[1],roi_size[0]))[...,np.newaxis]
        ae_output_bgr = np.clip( (ae_output[:,:,1:] + 1) * 255 / 2, 0, 255)
        ae_output_bgr = cv2.resize(ae_output_bgr, (roi_size[1],roi_size[0]))
        ae_output_masked = (ae_output_a/255 * ae_output_bgr + (1 - ae_output_a/255) * roi).astype('uint8') # BGR, [0, 255]
        '''
        p = np.hstack([cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype('uint8'), np.concatenate((ae_output_a, ae_output_a, ae_output_a), axis=-1).astype('uint8'), cv2.cvtColor(ae_output_bgr.astype('uint8'), cv2.COLOR_BGR2RGB),cv2.cvtColor(ae_output_masked.astype('uint8'), cv2.COLOR_BGR2RGB)])
        plt.imshow(p)
        exit()
        '''
        self.ae_output_a = ae_output_a         
        if color_correction == "adain":
            self.ae_output_masked = adain(ae_output_masked, roi)
            self.ae_output_bgr = adain(ae_output_bgr, roi)
        elif color_correction == "adain_xyz":
            self.ae_output_masked = adain(ae_output_masked, roi, color_space="XYZ")
            self.ae_output_bgr = adain(ae_output_bgr, roi, color_space="XYZ")
        elif color_correction == "hist_match":
            self.ae_output_masked = color_hist_match(ae_output_masked, roi)
            self.ae_output_bgr = color_hist_match(ae_output_bgr, roi)
        #elif color_correction == "seamless_clone":
        #    self.ae_output_masked = seamless_clone(ae_output_masked, roi)
        #    self.ae_output_bgr = seamless_clone(ae_output_bgr, roi)
        else:
            self.ae_output_masked = ae_output_masked
            self.ae_output_bgr = ae_output_bgr
    
    def _merge_img_and_mask(self, ae_output_bgr, ae_output_masked, edge_blur, input_size, roi, roi_coverage, color_correction):  
        blend_mask = self.get_feather_edges_mask(roi, roi_coverage, edge_blur)      
        blended_img = blend_mask/255 * ae_output_masked + (1-blend_mask/255) * roi
        roi_x0, roi_x1, roi_y0, roi_y1 = self.roi_bound
        result = self.img_bgr.copy()
        result[roi_x0:roi_x1, roi_y0:roi_y1,:] = blended_img 
        result_alpha = np.zeros_like(self.img_bgr)
        result_alpha[roi_x0:roi_x1, roi_y0:roi_y1,:] = (blend_mask/255) * self.ae_output_a
        result_rawRGB = self.img_bgr.copy()
        if color_correction == "seamless_clone":
            result_rawRGB = seamless_clone(ae_output_bgr, result_rawRGB, self.ae_output_a, roi_x0, roi_y0)
        else:
            result_rawRGB[roi_x0:roi_x1, roi_y0:roi_y1,:] = ae_output_bgr 
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 
        result_rawRGB = cv2.cvtColor(result_rawRGB, cv2.COLOR_BGR2RGB)
        self.result = result 
        self.result_rawRGB = result_rawRGB
        self.result_alpha = result_alpha
    
    @staticmethod
    def get_feather_edges_mask(img, roi_coverage, edge_blur=0):
        img_size = img.shape
        mask = np.zeros_like(img)
        roi_x0, roi_x1, roi_y0, roi_y1 = cal_roi(img_size, roi_coverage)
        mask[roi_x0:roi_x1, roi_y0:roi_y1,:]  = 255
        if edge_blur:
            mask = cv2.GaussianBlur(mask,(edge_blur,edge_blur),10)
        else:
            mask[:,:,:] = 255
        return mask  

    def transform(self, inp_img, inp_layout, direction, roi_coverage, color_correction, edge_blur, IMAGE_SHAPE):
        self.check_generator_model(self.model)
        self.check_roi_coverage(inp_img, roi_coverage)
        
        if direction == "AtoB":
            self.path_func = self.model.path_abgr_B
        elif direction == "BtoA":
            self.path_func = self.model.path_abgr_A
        else:
            raise ValueError(f"direction should be either AtoB or BtoA, recieved {direction}.")
        
        self.inp_img = inp_img
        self.inp_layout = inp_layout
        
        # pre-process input image
        # Set 5 members: self.img_bgr, self.input_size, self.roi, self.roi_size, self.ae_input
        self._preprocess_inp_img(self.inp_img, self.inp_layout, roi_coverage, IMAGE_SHAPE)

        # model inference
        # Set 1 member: self.ae_output
        self._ae_forward_pass(self.ae_input, self.ae_layout_input)
        
        # post-process transformed roi image
        # Set 3 members: self.ae_output_a, self.ae_output_masked, self.ae_output_bgr
        self._postprocess_roi_img(self.ae_output, self.roi, self.roi_size, color_correction)

        # merge transformed output back to input image
        # Set 3 members: self.result, self.result_rawRGB, self.result_alpha
        self._merge_img_and_mask(self.ae_output_bgr, self.ae_output_masked, edge_blur,
                                  self.input_size, self.roi, roi_coverage, color_correction)
        
        return self.result, self.result_rawRGB, self.result_alpha
    
    @staticmethod
    def check_generator_model(model):
        if model is None: 
            raise ValueError(f"Generator model has not been set.")
    
    @staticmethod
    def check_roi_coverage(inp_img, roi_coverage):
        input_size = inp_img.shape        
        roi_x, roi_y = int(input_size[0]*(1-roi_coverage)), int(input_size[1]*(1-roi_coverage))
        # roi = 0 stands for auto roi
        if roi_coverage == 0:
            return
        if roi_x == 0 or roi_y == 0:
            raise ValueError("Error occurs when cropping roi image. \
            Consider increasing min_face_area or decreasing roi_coverage.")
        