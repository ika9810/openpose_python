# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

import json
import numpy as np

rHandX = []
rHandY = []
rHandConf = []

lHandX = []
lHandY = []
lHandConf = []

poseX = []
poseY = []
poseConf = []

faceX = []
faceY = []
faceConf = []

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000241.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["face"] = True
    params["hand"] = True

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints[0]))
    print("Face keypoints: \n" + str(datum.faceKeypoints[0]))
    print("Left hand keypoints: \n" + str(datum.handKeypoints[0][0]))
    print("Right hand keypoints: \n" + str(datum.handKeypoints[1][0]))
    pose = datum.poseKeypoints[0]
    face = datum.faceKeypoints[0]
    lHand = datum.handKeypoints[0][0]
    rHand = datum.handKeypoints[1][0]

    for i in range(len(face)):       #얼굴
        faceX.append(face[i][0])
        faceY.append(face[i][1])
        faceConf.append(face[i][2])
    for i in range(len(lHand)):      #왼손
        lHandX.append(lHand[i][0])
        lHandY.append(lHand[i][1])
        lHandConf.append(lHand[i][2])
    for i in range(len(rHand)):      #오른손
        rHandX.append(rHand[i][0])
        rHandY.append(rHand[i][1])
        rHandConf.append(rHand[i][2])
    for i in ['0', '1', '2', '3', '4', '5', '6', '7', '14', '15', '16', '17']:      #body 몸
        poseX.append(pose[i][0])
        poseY.append(pose[i][1])
        poseConf.append(pose[i][2])

    rHandX = np.asarray(rHandX)
    rHandY = np.asarray(rHandY)
    rHandConf = np.asarray(rHandConf)
    lHandX = np.asarray(lHandX)
    lHandY = np.asarray(lHandY)
    lHandConf = np.asarray(lHandConf)

    poseX = np.asarray(poseX)
    poseY = np.asarray(poseY)
    poseConf = np.asarray(poseConf)

    faceX = np.asarray(faceX)
    faceY = np.asarray(faceY)
    faceConf = np.asarray(faceConf)

    res = np.concatenate([lHandX, lHandY, rHandX, rHandY, faceX, faceY, poseX, poseY])
    print(res)
    cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)
