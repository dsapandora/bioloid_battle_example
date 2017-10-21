#!/usr/bin/env python
import rospy
import random
from spencer_tracking_msgs.msg import TrackedPersons
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_json
import numpy as np
import tensorflow as tf
from numpy import argmax
import serial

#FILE
json_file = open('modelsoftmax_matrix.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model._make_predict_function()
graph = tf.get_default_graph()
# load weights into new model
loaded_model.load_weights("modelsoftmax_matrix.h5")
bioloid_action = ["WFWD\n","WBWD\n","WLT \n","WRT \n","WLSD\n","WRSD\n","WFLS\n","WFRS\n","WBLS\n","WBRS\n","WAL \n","WAR \n","WFLT\n","WFRT\n","WBLT\n","WBRT\n","WRDY\n","SIT \n","STND\n","ATKL\n","ATKR\n","ATKF\n", "ATKD\n"]


def reverse_map_prediction(prediction_matrix):
    # Para utilizar categorical_crossentropy, se tuvo
    # que transformar el arreglo de numeros en matrices, ahora necesitamos
    # regresarlo a su valor original para poder ser usado la lista bioloi
    index = argmax(prediction_matrix, axis=1) - 1
    return bioloid_action[index]

def callback(detectedPersons):
    output = "DETECTADOS: "
    if detectedPersons.tracks:
        for detectedPerson in detectedPersons.tracks:
            # The TrackSynchronizer invoking this callback guarantees that the detectedPersons message is buffered until a
            # track association is available for these detections (by comparing message timestamps of tracks and detections).
            detectionId = detectedPerson.track_id
            #
            pose_covariance = list(detectedPerson.pose.covariance)
            twist_covariance = list(detectedPerson.twist.covariance)
            sublist = [detectedPerson.pose.pose.position.x,detectedPerson.pose.pose.position.y,detectedPerson.pose.pose.position.z, detectedPerson.pose.pose.orientation.x, detectedPerson.pose.pose.orientation.y, detectedPerson.pose.pose.orientation.z]
            sublist.extend(pose_covariance)
            twist_list = [detectedPerson.twist.twist.linear.x, detectedPerson.twist.twist.linear.y, detectedPerson.twist.twist.linear.z]
            sublist.extend(twist_list)
            sublist.extend(twist_covariance)
            values = np.array([sublist])
            global graph
            global ser
            with graph.as_default():
                prediction_matrix = loaded_model.predict(values)
                value = reverse_map_prediction(prediction_matrix)
                output += str(detectionId) + ","+ value
                # ACTIVATE when you have the robot connected to the computer
                # The port can change.
                if detectionId == 0:
                    ser = serial.Serial('/dev/ttyUSB0', 57600)
                    ser.write(value)
                    ser.close()
                # if detectionId == 1:
                #     ser = serial.Serial('/dev/ttyUSB1', 57600)
                #     ser.write(value)
                #     ser.close()
    rospy.loginfo(output)

def main():
    rospy.init_node('robot_controller', anonymous=True)
    rospy.Subscriber("/spencer/perception/tracked_persons", TrackedPersons, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    main()
