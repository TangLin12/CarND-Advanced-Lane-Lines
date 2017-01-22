import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
class Calibrater():
	def __init__(self):
		self.load()

	def get_objp_and_img(self,path,n_corners=(9,6)):
		objp=np.zeros((n_corners[0]*n_corners[1],3),np.float32)
		
		objp[:,:2]=np.mgrid[0:n_corners[0], 0:n_corners[1]].T.reshape(-1,2)

		objpoints=[]
		imgpoints=[]

		images=glob.glob(path)

		for idx,fname in enumerate(images):
			img=cv2.imread(fname)
			gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

			ret,corners=cv2.findChessboardCorners(gray, n_corners, None)
			if ret ==True:
				objpoints.append(objp)
				imgpoints.append(corners)
		return objpoints,imgpoints
		
	def camera_calibration(self,path,n_corners=(9,6)):
		return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

	def load(self,path="wide_dist_pickle.p"):
		dist_pickle = pickle.load( open( path, "rb" ) )
		self.mtx=dist_pickle["mtx"]
		self.dist=dist_pickle["dist"]


	def save(self,path='wide_dist_pickle.p'):
		dist_pickle = {}
		dist_pickle["mtx"] = self.mtx
		dist_pickle["dist"] = self.dist
		pickle.dump( dist_pickle, open( path, "wb" ) )

	def dst_correction(self,img):
		return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
