#!/usr/bin/env python3

from __future__ import print_function
import rospy
from std_msgs.msg import String
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import threading

class rl_sub():
    def __init__(self):
        self.sub = rospy.Subscriber("chatter", String, self.callback)
        self.print_var = "s"

    def callback(self, data):
        rospy.loginfo("I heard %s" % data.data)
        # bridge = CvBridge()
        # rospy.loginfo("I heard 2")
        # int_values = [x for x in data.data]
        # dat = np.asarray(int_values, dtype=int)
        # print(np.where(dat==255)[0].shape)
        # try:
        #     cv_image = bridge.imgmsg_to_cv2(data, 'mono8')
        # except CvBridgeError as e:
        #     print(e)
        # rospy.loginfo("I save ")
        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(3)\
    
    def main(self):
        while not rospy.is_shutdown():
            pass
        rospy.loginfo("SHUTTTTTTT")
        return

def runner(): 
    while not rospy.is_shutdown():
        rospy.spin()

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('pic_listner', anonymous=True)
    th = threading.Thread(target=runner)
    th.daemon = True
    th.start()
    ic = rl_sub()
    ic.main()

    th.join()
    

    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()

if __name__ == '__main__':
    listener()