from kalman_filter import KalmanFilter
from landmarks_alignment import *
from color_correction import seamless_clone
from face_transformer import FaceTransformer
from vc_utils import *
from face_layout import FaceMarker
import numpy as np
import io
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


class VideoConverter(object):
    """
    This class is for video conversion
    
    Attributes:
        ftrans: FaceTransformer instance
        fdetect: MTCNNFaceDetector instance
        prev_x0, prev_x1, prev_y0, prev_y1, frames: Variables for smoothing bounding box
        kf0, kf1: KalmanFilter instances for smoothing bounding box
    """
    def __init__(self):        
        # Variables for smoothing bounding box
        self.prev_x0 = 0
        self.prev_x1 = 0
        self.prev_y0 = 0
        self.prev_y1 = 0        
        self.frames = 0
        
        # face transformer
        self.ftrans = FaceTransformer()

        # face marker
        self.fm = None
        # buf array to store layout in mem
        self.buf_store = []
        
        # MTCNN face detector
        self.fdetect = None
        
        # Kalman filters
        self.kf0 = None
        self.kf1 = None
        
    def set_gan_model(self, model):
        self.ftrans.set_model(model)
        
    def set_face_detector(self, fdetect):
        self.fdetect = fdetect
    
    def _get_smoothed_coord(self, x0, x1, y0, y1, img_shape, use_kalman_filter=True, ratio=0.65):
        if not use_kalman_filter:
            x0 = int(ratio * self.prev_x0 + (1-ratio) * x0)
            x1 = int(ratio * self.prev_x1 + (1-ratio) * x1)
            y1 = int(ratio * self.prev_y1 + (1-ratio) * y1)
            y0 = int(ratio * self.prev_y0 + (1-ratio) * y0)
        else:
            x0y0 = np.array([x0, y0]).astype(np.float32)
            x1y1 = np.array([x1, y1]).astype(np.float32)
            self.kf0.correct(x0y0)
            pred_x0y0 = self.kf0.predict()
            self.kf1.correct(x1y1)
            pred_x1y1 = self.kf1.predict()
            x0 = np.max([0, pred_x0y0[0][0]]).astype(np.int)
            x1 = np.min([img_shape[0], pred_x1y1[0][0]]).astype(np.int)
            y0 = np.max([0, pred_x0y0[1][0]]).astype(np.int)
            y1 = np.min([img_shape[1], pred_x1y1[1][0]]).astype(np.int)
            if x0 == x1 or y0 == y1:
                x0, y0, x1, y1 = self.prev_x0, self.prev_y0, self.prev_x1, self.prev_y1
        return x0, x1, y0, y1    
    
    def _set_prev_coord(self, x0, x1, y0, y1):
        self.prev_x0 = x0
        self.prev_x1 = x1
        self.prev_y1 = y1
        self.prev_y0 = y0
        
    def _init_kalman_filters(self, noise_coef):
        self.kf0 = KalmanFilter(noise_coef=noise_coef)
        self.kf1 = KalmanFilter(noise_coef=noise_coef)
        
    def _init_env(self, options):
        if options['use_kalman_filter']:
            self._init_kalman_filters(options["kf_noise_coef"])
        
        self.frames = 0
        self.prev_x0 = self.prev_x1 = self.prev_y0 = self.prev_y1 = 0

    def gene_layout(self, input_fn, options, duration=None):
        self.check_options(options)
        self._init_env(options)

        if self.fdetect is None:
            raise Exception(f"face detector has not been set through VideoConverter.set_face_detector() yet.")
        
        if self.fm is None:
            self.fm = FaceMarker()
        
        clip1 = VideoFileClip(input_fn)
        if type(duration) is tuple:
            clip = clip1.fl_image(lambda img: self.prepare_layout(img, options)).subclip(duration[0], duration[1])
        else:
            clip = clip1.fl_image(lambda img: self.prepare_layout(img, options))
        
        clip.write_videofile('tmp.mp4', audio=False)
        clip1.reader.close()
        
        print('{} layouts has been generated.'.format(len(self.buf_store)))

        return self.buf_store

    def convert(self, input_fn, input_buf, output_fn, options, duration=None):
        self.check_options(options)
        self._init_env(options)
        self.buf_store = input_buf

        if self.fdetect is None:
            raise Exception(f"face detector has not been set through VideoConverter.set_face_detector() yet.")
        
        clip1 = VideoFileClip(input_fn)
        if type(duration) is tuple:
            clip = clip1.fl_image(lambda img: self.process_video(img, options)).subclip(duration[0], duration[1])
        else:
            clip = clip1.fl_image(lambda img: self.process_video(img, options))
        clip.write_videofile(output_fn, audio=True)
        clip1.reader.close()
        try:
            clip1.audio.reader.close_proc()
        except:
            pass

    def prepare_layout(self, input_img, options):
        image = input_img

        # detect face using MTCNN (faces: face bbox coord, pnts: landmarks coord.)
        faces, pnts = self.fdetect.detect_face(image, minsize=20, 
                                               threshold=options["detec_threshold"], 
                                               factor=0.709, 
                                               use_auto_downscaling=options["use_auto_downscaling"],
                                               min_face_area=options["min_face_area"]
                                              )
        best_conf_score = 0
        # loop through all detected faces
        for i, (x0, y1, x1, y0, conf_score) in enumerate(faces):
            lms = pnts[:,i:i+1]
            # smoothe the bounding box
            if options["use_smoothed_bbox"]:
                if self.frames != 0 and conf_score >= best_conf_score:
                    x0, x1, y0, y1 = self._get_smoothed_coord(
                        x0, x1, y0, y1, 
                        img_shape=image.shape, 
                        use_kalman_filter=options["use_kalman_filter"],
                        ratio=options["bbox_moving_avg_coef"],
                    )
                    self._set_prev_coord(x0, x1, y0, y1)
                    best_conf_score = conf_score
                    self.frames += 1
                elif conf_score <= best_conf_score:
                    self.frames += 1
                else:
                    if conf_score >= best_conf_score:
                        self._set_prev_coord(x0, x1, y0, y1)
                        best_conf_score = conf_score
                    if options["use_kalman_filter"]:
                        for i in range(200):
                            self.kf0.predict()
                            self.kf1.predict()
                    self.frames += 1
            

            # generate face layout
            det_face_im = input_img[int(x0):int(x1),int(y0):int(y1),:]
            layout = self.fm.mark(det_face_im)
            buf = io.BytesIO()
            plt.imsave(buf, arr=layout, format='jpg')
            buf.seek(0)
            self.buf_store.append(buf)

        return image

        
    def process_video(self, input_img, options): 
        """Transform detected faces in single input frame."""
        image = input_img

        # detect face using MTCNN (faces: face bbox coord, pnts: landmarks coord.)
        faces, pnts = self.fdetect.detect_face(image, minsize=20, 
                                               threshold=options["detec_threshold"], 
                                               factor=0.709, 
                                               use_auto_downscaling=options["use_auto_downscaling"],
                                               min_face_area=options["min_face_area"]
                                              )

        # check if any face detected
        if len(faces) == 0:
            comb_img = get_init_comb_img(input_img)
            triple_img = get_init_triple_img(input_img, no_face=True)

        # init. output image
        mask_map = get_init_mask_map(image)
        comb_img = get_init_comb_img(input_img)
        best_conf_score = 0

        # loop through all detected faces
        for i, (x0, y1, x1, y0, conf_score) in enumerate(faces):   
            lms = pnts[:,i:i+1]
            # smoothe the bounding box
            if options["use_smoothed_bbox"]:
                if self.frames != 0 and conf_score >= best_conf_score:
                    x0, x1, y0, y1 = self._get_smoothed_coord(
                        x0, x1, y0, y1, 
                        img_shape=image.shape, 
                        use_kalman_filter=options["use_kalman_filter"],
                        ratio=options["bbox_moving_avg_coef"],
                    )
                    self._set_prev_coord(x0, x1, y0, y1)
                    best_conf_score = conf_score
                    self.frames += 1
                elif conf_score <= best_conf_score:
                    self.frames += 1
                else:
                    if conf_score >= best_conf_score:
                        self._set_prev_coord(x0, x1, y0, y1)
                        best_conf_score = conf_score
                    if options["use_kalman_filter"]:
                        for i in range(200):
                            self.kf0.predict()
                            self.kf1.predict()
                    self.frames += 1
            
            # transform face
            # get detected face
            det_face_im = input_img[int(x0):int(x1),int(y0):int(y1),:]

            buf = self.buf_store.pop(0)
            layout = plt.imread(buf, format='jpg')
            buf.close()
            det_face_layout = layout
            try:
                # get src/tar landmarks
                src_landmarks = get_src_landmarks(x0, x1, y0, y1, lms)
                tar_landmarks = get_tar_landmarks(det_face_im)

                # align detected face
                aligned_det_face_im = landmarks_match_mtcnn(det_face_im, src_landmarks, tar_landmarks)
                aligned_det_face_layout = landmarks_match_mtcnn(det_face_layout, src_landmarks, tar_landmarks)

                # face transform
                r_im, r_rgb, r_a = self.ftrans.transform(
                    aligned_det_face_im,
                    aligned_det_face_layout, 
                    direction=options["direction"], 
                    roi_coverage=options["roi_coverage"],
                    color_correction=options["use_color_correction"],
                    edge_blur=options["edge_blur"],
                    IMAGE_SHAPE=options["IMAGE_SHAPE"]
                    )

                # reverse alignment
                rev_aligned_det_face_im = landmarks_match_mtcnn(r_im, tar_landmarks, src_landmarks)
                rev_aligned_det_face_im_rgb = landmarks_match_mtcnn(r_rgb, tar_landmarks, src_landmarks)
                rev_aligned_mask = landmarks_match_mtcnn(r_a, tar_landmarks, src_landmarks)

                # merge source face and transformed face
                result = np.zeros_like(det_face_im)
                result = rev_aligned_mask/255*rev_aligned_det_face_im_rgb + (1-rev_aligned_mask/255)*det_face_im
                result_a = rev_aligned_mask
            except:            
                # catch exceptions for landmarks alignment errors (if any)
                print(f"Face alignment error occurs at frame {self.frames}.")
                
                result, _, result_a = self.ftrans.transform(
                    det_face_im,
                    det_face_layout,
                    direction=options["direction"], 
                    roi_coverage=options["roi_coverage"],
                    color_correction=options["use_color_correction"],
                    edge_blur=options["edge_blur"],
                    IMAGE_SHAPE=options["IMAGE_SHAPE"]
                    )
            #if options["use_color_correction"] == "seamless_clone":
                #syn_face = seamless_clone(result, input_img, result_a, x0, y0)
                #comb_img[:, input_img.shape[1]:input_img.shape[1]*2, :] = syn_face
            #else:
            comb_img[int(x0):int(x1),input_img.shape[1]+int(y0):input_img.shape[1]+int(y1),:] = result
            
            # Enhance output
            if options["enhance"] != 0:
                comb_img = -1*options["enhance"] * get_init_comb_img(input_img) + (1+options["enhance"]) * comb_img
                comb_img = np.clip(comb_img, 0, 255)              
            
            if conf_score >= best_conf_score:
                mask_map[int(x0):int(x1),int(y0):int(y1),:] = result_a
                mask_map = np.clip(mask_map + .15 * input_img, 0, 255)   
                # Possible bug: when small faces are detected before the most confident face,
                #               the mask_map will show brighter input_img
            else:
                mask_map[int(x0):int(x1),int(y0):int(y1),:] += result_a
                mask_map = np.clip(mask_map, 0, 255)

            triple_img = get_init_triple_img(input_img)
            triple_img[:, :input_img.shape[1]*2, :] = comb_img
            triple_img[:, input_img.shape[1]*2:, :] = mask_map
        
        if options["output_type"] == 1:
            return comb_img[:, input_img.shape[1]:, :]  # return only result image
        elif options["output_type"] == 2:
            return comb_img  # return input and result image combined as one
        elif options["output_type"] == 3:
            return triple_img #return input,result and mask heatmap image combined as one
        
    @staticmethod
    def check_options(options):
        if options["roi_coverage"] < 0 or options["roi_coverage"] >= 1:
            raise ValueError(f"roi_coverage should be between 0 and 1 (exclusive). 0 for auto roi.")
        if options["bbox_moving_avg_coef"] < 0 or options["bbox_moving_avg_coef"] > 1:
            raise ValueError(f"bbox_moving_avg_coef should be between 0 and 1 (inclusive).")
        if options["detec_threshold"] < 0 or options["detec_threshold"] > 1:
            raise ValueError(f"detec_threshold should be between 0 and 1 (inclusive).")
        if options["use_smoothed_bbox"] not in [True, False]:
            raise ValueError(f"use_smoothed_bbox should be a boolean.")
        if options["use_kalman_filter"] not in [True, False]:
            raise ValueError(f"use_kalman_filter should be a boolean.")
        if options["use_auto_downscaling"] not in [True, False]:
            raise ValueError(f"use_auto_downscaling should be a boolean.")
        if options["output_type"] not in range(1,4):
            ot = options["output_type"]
            raise ValueError(f"Received an unknown output_type option: {ot}.")
