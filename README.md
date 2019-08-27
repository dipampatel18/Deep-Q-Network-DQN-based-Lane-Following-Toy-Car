# Deep-Q-Network-DQN-based-Lane-Following-Toy-Car

This project is an attempt to present a reinforcement learning approach using Deep Q-Networks to steer a toy car on a real track made of white road and black lanes. An action-based reward function is proposed, which is motivated by a potential use in real world reinforcement learning scenarios. We provide a general and easy to obtain reward: the distance travelled by the car without the supervisor taking control. Compared to a naive distance-based reward function, it improves the overall driving behavior of the car agent. From randomly initialized parameters, our model is trying to learn a policy for lane following using a single monocular image as input. We use the deep q-network algorithm, with all exploration and optimization performed on the computer connected to the raspberry pi. This demonstrates a new framework for autonomous driving which moves away from reliance on defined logical rules and mapping. We discuss the challenges and opportunities to scale this approach to a broader range of autonomous driving tasks.

<p align="center">
  <img width="700" height="380" src="/files/RaceTrack.jpg">
</p>

### Approach

Using only the monocular camera image as input, the car is trained to follow the lanes without any previously collected data for training. The algorithm is rewarded based on the distance travelled by the car before it leaves the lane and is stopped by the supervisor by giving a negative reward. A model-free approach based on the trial and error strategy where it keeps updating its knowledge.

<p align="center">
  <img width="700" height="380" src="/files/TestRun.gif">
</p>

[Link](https://drive.google.com/file/d/1rllwzH8UelCLR_YSGW9ly_s5jMo-VzgR/view) to Entire Testing Video

<p align="center">
  <img width="700" height="380" src="/files/TrainRun.gif">
</p>

[Link](https://drive.google.com/file/d/180Qn422pd3R9GoEO2Z3jaWm0X48zOT9E/view) to Entire Training Video 

From the randomly initialized parameters, the model is trying to learn a policy for lane following and optimizing it on the go. Raspberry Pi 3B+ was used as an onboard processor on the car to stream frames and perform the actions (forward, left and right) received from the computer which acted as the central processor. ROS was used for communication between Raspberry Pi and Computer. Also, Tensorflow 1.8.0 framework was used to train the convolutional neural network

<p align="center">
  <img width="700" height="380" src="/files/ToyCar.jpg">
</p>

The hardware used for the project was-

- RC Toy Car
- Raspberry Pi 3B+
- Raspberry Pi Camera V2.1
- L293D Motor Driver
- Cables
- Batteries
- White Paper as Track and Black Tape as Lanes

<p align="center">
  <img width="700" height="380" src="/files/FPVTrack.jpg">
</p>

[Link](https://drive.google.com/open?id=1Iq9u1Ckv_Lp9fQoctYUTdmuSmczXdpai) to the Code Package

See the Full Report for the detailed description and the Presentation for a quick overview
