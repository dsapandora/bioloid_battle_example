#!/usr/bin/env python
import roslib; roslib.load_manifest('robot_battle')
import rospy
from std_msgs.msg import String
import serial

def talker():
    ser = serial.Serial('/dev/ttyUSB0', 57600)

    pub = rospy.Publisher('/serial_msg', String, queue_size=10)
    rospy.init_node('serial_talker')
    while not rospy.is_shutdown():
       data= ser.read(2)
       rospy.loginfo(data)
       pub.publish(String(data))


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
