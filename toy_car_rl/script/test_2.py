#!/usr/bin/env python3
# license removed for brevity
import rospy
from std_msgs.msg import String
# from cv_bridge import CvBridge, CvBridgeError
from toy_car_rl.srv import *
import threading

class talk():
    def __init__(self):
        self.pub = rospy.Publisher('chatter', String, queue_size=10)
        self.srv = rospy.Service("rwd_srv", rwd, self.reward_change)
        self.print_var = "shivang"

    def talker(self):        
        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            # hello_str = "hello world %s" % self.print_var
            hello_str = "I AM IRON MAN "
            # rospy.loginfo(hello_str)
            # rate.sleep()
            self.pub.publish(hello_str[2:4])
            # rate.sleep()
            self.pub.publish(hello_str[5:9])
            # rate.sleep()
            self.pub.publish(hello_str[10:13])
            rate.sleep()

    def reward_change(self, data):
        self.print_var = data.rw_type
        return rwdResponse(True)

def runner(): 
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('talker', anonymous=True)
    th = threading.Thread(target=runner)
    th.daemon = True
    th.start()
    tk = talk()
    tk.talker()
    th.join()       
    

#!/usr/bin/env python3

from __future__ import print_function
import rospy
from std_msgs.msg import String
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class rl_sub():
    def __init__(self):
        self.sub = rospy.Subscriber("picam_image", Image, self.callback)

    def callback(self, data):
        rospy.loginfo("I heard ")
        bridge = CvBridge()
        rospy.loginfo("I heard 2")
        int_values = [x for x in data.data]
        dat = np.asarray(int_values, dtype=int)
        print(np.where(dat==255)[0].shape)
        # try:
        #     cv_image = bridge.imgmsg_to_cv2(data, 'mono8')
        # except CvBridgeError as e:
        #     print(e)
        # rospy.loginfo("I save ")
        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(3)
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('pic_listner', anonymous=True)
    ic = rl_sub()
    

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()