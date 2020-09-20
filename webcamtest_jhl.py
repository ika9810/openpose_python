# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

import json
import numpy as np

Frames = list()

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + \
                '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.ahttp://220.149.86.25:8200/mjpegppend('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
    # webcam test jhl
    # Flags
    # ../../../examples/media/COCO_val2014_000000000241.jpg
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000241.jpg",
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["face"] = True
    params["hand"] = True

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1:
            next_item = args[1][i+1]
        else:
           next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:
                params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params:
                params[key] = next_item
    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    frames = 0
    while(1):
        cap = cv2.VideoCapture('http://220.149.86.25:8200/jpeg')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        error, dst = cap.read()
        if not error:
            continue
        frames += 1
        # Process Image
        datum = op.Datum()
        #args[0].image_path
        #imageToProcess = cv2.imread(args[0].image_path)
        #imageToProcess = dst
        #datum.cvInputData = imageToProcess
        datum.cvInputData = dst
        opWrapper.emplaceAndPop([datum])

        # Display Image
        #print("Body keypoints: \n" + str(datum.poseKeypoints[0]))
        #print("Face keypoints: \n" + str(datum.faceKeypoints[0]))
        #print("Left hand keypoints: \n" + str(datum.handKeypoints[0][0]))
        #print("Right hand keypoints: \n" + str(datum.handKeypoints[1][0]))
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

        #jhl_part
        pose = datum.poseKeypoints[0]
        face = datum.faceKeypoints[0]
        lHand = datum.handKeypoints[0][0]
        rHand = datum.handKeypoints[1][0]
        # 리스트에 넣어주기
        for i in range(len(face)):  # 얼굴
            faceX.append(face[i][0])
            faceY.append(face[i][1])
            faceConf.append(face[i][2])
        for i in range(len(lHand)):  # 왼손
            lHandX.append(lHand[i][0])
            lHandY.append(lHand[i][1])
            lHandConf.append(lHand[i][2])
        for i in range(len(rHand)):  # 오른손
            rHandX.append(rHand[i][0])
            rHandY.append(rHand[i][1])
            rHandConf.append(rHand[i][2])
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]:  # body 몸
            poseX.append(pose[i][0])
            poseY.append(pose[i][1])
            poseConf.append(pose[i][2])
        #print(frames)
        #numpy 형태
        _rHandX = np.asarray(rHandX)
        _rHandY = np.asarray(rHandY)
        _rHandConf = np.asarray(rHandConf)
        _lHandX = np.asarray(lHandX)
        _lHandY = np.asarray(lHandY)
        _lHandConf = np.asarray(lHandConf)

        _poseX = np.asarray(poseX)
        _poseY = np.asarray(poseY)
        _poseConf = np.asarray(poseConf)

        _faceX = np.asarray(faceX)
        _faceY = np.asarray(faceY)
        _faceConf = np.asarray(faceConf)

        res = np.concatenate([_lHandX, _lHandY, _rHandX, _rHandY, _faceX, _faceY, _poseX, _poseY])
	
        if len(Frames) >= 20:
            Frames.pop(0)
        Frames.append(res)
        print(len(Frames))
        result = np.asarray(Frames)
        print(result)
        #print(res)

        #cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
        if cv2.waitKey(1) == 27:
            break
except Exception as e:
    print("* ERROR ")
    print(e)
    sys.exit(-1)
