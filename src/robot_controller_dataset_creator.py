#!/usr/bin/env python
import rospy
import csv
import itertools
import random
from spencer_tracking_msgs.msg import TrackedPersons

def callback(detectedPersons):
    output = "DETECTADOS: "
    if detectedPersons.tracks:
        with open('/home/dsapandora/tracking_person.csv', 'a') as f:
            writer=csv.writer(f)
            for detectedPerson in detectedPersons.tracks:
                # The TrackSynchronizer invoking this callback guarantees that the detectedPersons message is buffered until a
                # track association is available for these detections (by comparing message timestamps of tracks and detections).
                detectionId = detectedPerson.track_id
                pose_covariance = list(detectedPerson.pose.covariance)
                twist_covariance = list(detectedPerson.twist.covariance)
                sublist = [detectedPerson.pose.pose.position.x,detectedPerson.pose.pose.position.y,detectedPerson.pose.pose.position.z, detectedPerson.pose.pose.orientation.x, detectedPerson.pose.pose.orientation.y, detectedPerson.pose.pose.orientation.z]
                sublist.extend(pose_covariance)
                twist_list = [detectedPerson.twist.twist.linear.x, detectedPerson.twist.twist.linear.y, detectedPerson.twist.twist.linear.z]
                sublist.extend(twist_list)
                sublist.extend(twist_covariance)
                sublist.append(random.uniform(0,22))
                writer.writerow(sublist)
    else:
        output += "Empty set of detections!"
    rospy.loginfo(output)

def main():
    rospy.init_node('robot_controller', anonymous=True)
    rospy.Subscriber("/spencer/perception/tracked_persons", TrackedPersons, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    main()
