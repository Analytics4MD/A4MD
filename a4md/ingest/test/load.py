#!/usr/bin/python

import sys
import numpy as np
from numpy import linalg as npALG
import math
import os.path
import os
import time
import pickle
from threading import Thread

def extract_frame(fileName, position):
	"""
	Keep extract frame information in trajectory file
	
	Parameters
	------------
	fileName: string 
			Trajectory file path
	logName : string
			Log file path


	Returns
	------------
	num : int
			Number of atoms in frame
	CA  : list of tuples
			Set of atom positions

	"""
	file = None
	while file is None:
		try:
			file = open(fileName)
		except:
			pass
	
	try:
		#lp = pickle.load( open( logName, "rb" ) )
		file.seek(position)
	except:
		file.seek(0)
	
	x_cords = []
	y_cords = []
	z_cords = []
	types = []
	#CA = []
	num = 0
	while True:
		line = file.readline()
		if not line:
			time.sleep(0.1)
			continue
		# TODO: set timer for exit

		elif line[:4]=="ATOM":
			if line[12:16]==" CA ":
				alternate_location = line[16]
				if alternate_location==' ' or alternate_location=='A':
					#CA.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
					x_cords.append(float(line[30:38]))
					y_cords.append(float(line[38:46]))
					z_cords.append(float(line[46:54]))
					types.append(int(line[22:26]))
					num += 1

		elif line[:6]=="ENDMDL":
			break

		elif line[:3]=="END":
			# Repeat searching from beginning  
			file.seek(0)
			num = -1

			#break 

	lp = file.tell()
	#pickle.dump(lp, open( logName, "wb" ))
	return [types, x_cords, y_cords, z_cords, None, None, None, None, None, None, None, lp]

