########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np

import datetime


if __name__ == "__main__":
	print("Running Body Tracking sample ... Press 'q' to quit")

    # Create a Camera object
	zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
	init_params = sl.InitParameters()
	init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
	init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
	init_params.depth_mode = sl.DEPTH_MODE.ULTRA
	init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
	if len(sys.argv) == 2:
		filepath = sys.argv[1]
		print("Using SVO file: {0}".format(filepath))
		init_params.svo_real_time_mode = True
		init_params.set_from_svo_file(filepath)

    # Open the camera
	err = zed.open(init_params)
	if err != sl.ERROR_CODE.SUCCESS:
		exit(1)

	# Enable Positional tracking (mandatory for object detection)
	positional_tracking_parameters = sl.PositionalTrackingParameters()
	# If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
	# positional_tracking_parameters.set_as_static = True
	zed.enable_positional_tracking(positional_tracking_parameters)

	obj_param = sl.ObjectDetectionParameters()
	obj_param.enable_body_fitting = True            # Smooth skeleton move
	obj_param.enable_tracking = True                # Track people across images flow
	obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST 
	obj_param.body_format = sl.BODY_FORMAT.POSE_18  # Choose the BODY_FORMAT you wish to use

	# Enable Object Detection module
	zed.enable_object_detection(obj_param)

	obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
	obj_runtime_param.detection_confidence_threshold = 40

	# Get ZED camera information
	camera_info = zed.get_camera_information()

    # 2D viewer utilities
	display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280), min(camera_info.camera_resolution.height, 720))
	image_scale = [display_resolution.width / camera_info.camera_resolution.width
		     , display_resolution.height / camera_info.camera_resolution.height]

	# Create OpenGL viewer
	viewer = gl.GLViewer()
	viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking,obj_param.body_format)

	# Create ZED objects filled in the main loop
	bodies = sl.Objects()
	image = sl.Mat()

	#open txt file
	dt_now = datetime.datetime.now()
	date = dt_now.strftime('%y_%m_%d_%H_%M_%s')
	name = str('data/' + date)
	txt = open(name, 'a', encoding='utf-8', newline='\n')
	
	
		
	

	while viewer.is_available():
        # Grab an image
		if zed.grab() == sl.ERROR_CODE.SUCCESS:
			# Retrieve left image
			zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
			# Retrieve objects
			zed.retrieve_objects(bodies, obj_runtime_param)


			obj_array = bodies.object_list
			print(str(len(obj_array)) + " Person(s) detected\n")
			txt.write("\n\n\n" + str(len(obj_array)) + " Person(s) detected\n")
			if len(obj_array) > 0:
				n = 0
				while len(obj_array) > n:
					first_object = obj_array[n]
					print(str(n) + " th Person attributes:")
					txt.write("\n" + str(n + 1) + " th Person attributes:\n\n")
					print(" Confidence (" + str(int(first_object.confidence)) + "/100)")
					txt.write(" Confidence (" + str(int(first_object.confidence)) + "/100)\n")
					if obj_param.enable_tracking:
						print(" Tracking ID: " + str(int(first_object.id)) + " tracking state: " + repr(first_object.tracking_state) + " / " + repr(first_object.action_state))
						txt.write(" Tracking ID: " + str(int(first_object.id)) + " tracking state: " + repr(first_object.tracking_state) + " / " + repr(first_object.action_state) + "\n")
					position = first_object.position
					velocity = first_object.velocity
					dimensions = first_object.dimensions
					print(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]".format(position[0], position[1], position[2], velocity[0], velocity[1], velocity[2], dimensions[0],dimensions[1], dimensions[2]))
					txt.write(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]\n".format(position[0], position[1], position[2], velocity[0], velocity[1], velocity[2], dimensions[0],dimensions[1], dimensions[2]))
					if first_object.mask.is_init():
						print(" 2D mask available")
						txt.write(" 2D mask available\n")

					print(" Keypoint 2D ")
					txt.write(" Keypoint 2D \n")
					keypoint_2d = first_object.keypoint_2d
					for it in keypoint_2d:
						print("    " + str(it))
						txt.write("    " + str(it) + "\n")
					print("\n Keypoint 3D ")
					txt.write("\n Keypoint 3D \n")
					keypoint = first_object.keypoint
					for it in keypoint:
						print("    " + str(it))
						txt.write("    " + str(it) + "\n")
					
					n+=1

				#input('\nPress enter to continue: ')

			

			# Update GL view
			viewer.update_view(image, bodies) 
			# Update OCV view
			image_left_ocv = image.get_data()
			cv_viewer.render_2D(image_left_ocv,image_scale,bodies.object_list, obj_param.enable_tracking, obj_param.body_format)
			cv2.imshow("ZED | 2D View", image_left_ocv)
			cv2.waitKey(10)


			

	viewer.exit()
	
	txt.close()

	image.free(sl.MEM.CPU)
	# Disable modules and close camera
	zed.disable_object_detection()
	zed.disable_positional_tracking()
	zed.close()
