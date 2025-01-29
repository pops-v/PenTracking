import numpy as np
import cv2
import glob
import os

class StereoCalibration:
    def __init__(self, checkerboard_size=(9, 6), square_size=0.025):
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        self.objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2) * square_size
        
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []
    
    def capture_calibration_images(self, save_dir='calibration_images'):
        """Capture calibration images for stereo setup"""
        cap_left = cv2.VideoCapture(0)
        cap_right = cv2.VideoCapture(1)
        
        os.makedirs(f'{save_dir}/left', exist_ok=True)
        os.makedirs(f'{save_dir}/right', exist_ok=True)
        
        img_count = 0
        while img_count < 20:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            
            if not (ret_left and ret_right):
                break
            
            # Display captured frames
            display_left = frame_left.copy()
            display_right = frame_right.copy()
            
            cv2.putText(display_left, f"Images Captured: {img_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_right, f"Images Captured: {img_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Left Camera', display_left)
            cv2.imshow('Right Camera', display_right)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Save images
                cv2.imwrite(f'{save_dir}/left/left_{img_count}.jpg', frame_left)
                cv2.imwrite(f'{save_dir}/right/right_{img_count}.jpg', frame_right)
                img_count += 1
            
            elif key == ord('q'):
                break
        
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
    
    def find_checkerboard_corners(self, images_left, images_right):
        for img_left, img_right in zip(images_left, images_right):
            left = cv2.imread(img_left)
            right = cv2.imread(img_right)
            
            gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.checkerboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.checkerboard_size, None)
            
            if ret_left and ret_right:
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), self.criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), self.criteria)
                
                self.objpoints.append(self.objp)
                self.imgpoints_left.append(corners_left)
                self.imgpoints_right.append(corners_right)
    
    def calibrate_stereo_cameras(self):
        # Use the first image to get image dimensions
        first_left_img = cv2.imread(glob.glob('calibration_images/left/*.jpg')[0])
        first_right_img = cv2.imread(glob.glob('calibration_images/right/*.jpg')[0])
        
        gray_left = cv2.cvtColor(first_left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(first_right_img, cv2.COLOR_BGR2GRAY)

        # Calibrate individual cameras
        ret_left, mtx_left, dist_left, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, (gray_left.shape[1], gray_left.shape[0]), None, None
        )
        ret_right, mtx_right, dist_right, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, (gray_right.shape[1], gray_right.shape[0]), None, None
        )
        
        # Stereo calibration
        retval, camera_matrix_left, dist_coeffs_left, \
        camera_matrix_right, dist_coeffs_right, \
        rot_matrix, trans_vector, _, _ = cv2.stereoCalibrate(
            self.objpoints, 
            self.imgpoints_left, 
            self.imgpoints_right, 
            mtx_left, dist_left, 
            mtx_right, dist_right, 
            (gray_left.shape[1], gray_left.shape[0])
        )
        
        # Compute rectification
        rect_left, rect_right, proj_matrix_left, proj_matrix_right, \
        disparity_to_depth_map, roi_left, roi_right = cv2.stereoRectify(
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            (gray_left.shape[1], gray_left.shape[0]), 
            rot_matrix, trans_vector
        )
        
        return {
            'camera_matrix_left': camera_matrix_left,
            'dist_coeffs_left': dist_coeffs_left,
            'camera_matrix_right': camera_matrix_right,
            'dist_coeffs_right': dist_coeffs_right,
            'rectification_left': rect_left,
            'rectification_right': rect_right,
            'projection_matrix_left': proj_matrix_left,
            'projection_matrix_right': proj_matrix_right
        }
    
    def compute_disparity_map(self, left_img, right_img):
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=11,
            P1=8 * 3 * 11 ** 2,
            P2=32 * 3 * 11 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=50,
            speckleRange=2
        )
        
        disparity = stereo.compute(gray_left, gray_right)
        return disparity
    
    def compute_depth_map(self, disparity_map, focal_length, baseline):
        depth_map = np.zeros_like(disparity_map, dtype=np.float32)
        depth_map[disparity_map > 0] = focal_length * baseline / disparity_map[disparity_map > 0]
        return depth_map

def main():
    # Create calibration object
    calibrator = StereoCalibration()
    
    # Option to capture new calibration images
    capture_new = input("Capture new calibration images? (y/n): ").lower() == 'y'
    
    if capture_new:
        calibrator.capture_calibration_images()
    
    # Load calibration images
    images_left = sorted(glob.glob('calibration_images/left/*.jpg'))
    images_right = sorted(glob.glob('calibration_images/right/*.jpg'))
    
    # Find checkerboard corners
    calibrator.find_checkerboard_corners(images_left, images_right)
    
    # Calibrate cameras
    calibration_params = calibrator.calibrate_stereo_cameras()
    
    # Capture live video for depth estimation
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)
    
    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not (ret_left and ret_right):
            break
        
        # Compute disparity map
        disparity_map = calibrator.compute_disparity_map(frame_left, frame_right)
        
        # Compute depth map (adjust focal_length and baseline)
        depth_map = calibrator.compute_depth_map(
            disparity_map, 
            focal_length=500,  # Calibrate this value
            baseline=0.1  # Distance between cameras in meters
        )
        
        # Normalize depth map for visualization
        normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Visualization
        cv2.imshow('Left Camera', frame_left)
        cv2.imshow('Right Camera', frame_right)
        cv2.imshow('Depth Map', normalized_depth)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()