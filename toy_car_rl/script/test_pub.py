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
        # self.srv = rospy.Service("rwd_srv", rwd, self.reward_change)
        # self.print_var = "shivang"

    def talker(self):        
        rate = rospy.Rate(10)
        # while not rospy.is_shutdown():
        hello_str = "I AM IRON MAN "
        self.pub.publish(hello_str[2:4])
        self.pub.publish(hello_str[5:9])
        self.pub.publish(hello_str[10:13])
        rate.sleep()

    # def reward_change(self, data):
    #     self.print_var = data.rw_type
    #     return rwdResponse(True)

# def runner(): 
#     while not rospy.is_shutdown():
#         rospy.spin()

if __name__ == '__main__':
    rospy.init_node('talker', anonymous=True)
    tk = talk()
    tk.talker()   
    

