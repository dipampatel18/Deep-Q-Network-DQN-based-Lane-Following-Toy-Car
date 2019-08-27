# Deep-Q-Network-DQN-based-Lane-Following-Toy-Car

This project is an attempt to present a reinforcement learning approach using Deep Q-Networks to steer a toy car on a real track made of white road and black lanes. An action-based reward function is proposed, which is motivated by a potential use in real world reinforcement learning scenarios. We provide a general and easy to obtain reward: the distance travelled by the car without the supervisor taking control. Compared to a naive distance-based reward function, it improves the overall driving behavior of the car agent. From randomly initialized parameters, our model is trying to learn a policy for lane following using a single monocular image as input. We use the deep q-network algorithm, with all exploration and optimization performed on the computer connected to the raspberry pi. This demonstrates a new framework for autonomous driving which moves away from reliance on defined logical rules and mapping. We discuss the challenges and opportunities to scale this approach to a broader range of autonomous driving tasks.

<p align="center">
  <img width="700" height="380" src="/files/TestRun.gif">
</p>

<p align="center">
  <img width="700" height="380" src="/files/TrainRun.gif">
</p>
