import cv2
import numpy as np
import os

class Deprojector:
    
    def __init__(self):
        
        # self.intrinsics = self.loadCameraIntrinsics('webcam_acer')
        # self.distortion = self.loadCameraDistortion('webcam_acer')

        self.intrinsics = np.array([[914.0999755859375, 0, 637.8196411132812],
                                    [0, 914.7161254882812, 370.6839904785156],
                                    [0, 0, 1]])
        self.distortion = np.array([0.0,0.0,0.0,0.0,0.0])
        
    def loadCameraIntrinsics(self, camera_str):
        
        fn = camera_str + '_intrinsics.npy'
        path = os.path.join('src', 'cameras', fn)
        with open(path, 'rb') as f:
            mtx = np.load(f)
        
        return mtx
    
    def loadCameraDistortion(self, camera_str):
        
        fn = camera_str + '_distortion.npy'
        path = os.path.join('src', 'cameras', fn)
        with open(path, 'rb') as f:
            dist = np.load(f)
        
        return dist
    
    def deprojectPose(self, keypoints2D, keypoints3D_local):
        
        # Remove nose from keypoints3D
        keypoints3D_local = np.delete(keypoints3D_local, 9, 0)
        
        # Guess the translation
        tvec_estimative = np.array([[0, 0, 2000]], dtype = np.float64)
        rvec_estimative = np.array([[0,0,0]], dtype = np.float64)

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv2.solvePnP(keypoints3D_local, keypoints2D, self.intrinsics, self.distortion,
                                                        rvec=rvec_estimative ,tvec=tvec_estimative, useExtrinsicGuess=True,)
        
        rotation_matrix, jac = cv2.Rodrigues(rvecs)
        
        # Rotate and Translate points
        kpts_global = np.dot(rotation_matrix,keypoints3D_local.T) + tvecs.T
        
        # Result will be (3,16) because (3,3)*(16,3).T + (3,1) = (3,16)
        # So, we need to transpose kpts
        kpts_global = kpts_global.T

        # Project global kpts to image plane 
        tvecs_zeros = np.array([0.0, 0.0, 0.0]).T
        rvecs_zeros = np.array([0.0, 0.0, 0.0]).T
        posepoints, jac = cv2.projectPoints(kpts_global, rvecs_zeros, tvecs_zeros, self.intrinsics, self.distortion)
        
        homogeneous = np.eye(4)
        homogeneous[:3,:3] = rotation_matrix
        homogeneous[:3, 3] = tvecs.ravel()
        

        return kpts_global, posepoints, homogeneous
    
    def makePersonAxis(self, homogeneous):

        axis = np.float32([[300,0,0], [0,300,0], [0,0,-300]]).reshape(-1,3)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, homogeneous[:3,:3], homogeneous[:3,3], self.intrinsics, self.distortion)
        
        return imgpts