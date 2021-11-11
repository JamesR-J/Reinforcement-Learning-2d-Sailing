# Reinforcement-Learning-2d-Sailing
Inspired by Quantum Blacks use of RL in the Americas Cup I designed a simple sailing game with a basic loop course. Velocity of the boat was set by a polar curve function which depended on the angle of the boat to the wind, this was inspired by this project [AI Learns to Sail Upwind](https://ppierzc.github.io/ai-learns-to-sail-upwind/) but futher adapted to create a more realistic polar curve for reaching and downwind sailing.

Next I started to develop a Deep Q Network to train the boat. It used 5 "LIDAR" Rays to detect the edge of the sailing area, and these combined with the angle of the boat to the wind, the boats velocity, and the angle of the boat to the next gate as the state spaces. Reward gates were added as areas for the agent to aim for to go around the small black marks. The boat itself had three actions, turn left, turn right, and do nothing. Since the boats velocity was controlled by the direction to the wind it meant this was uncontrollable.

## 1st Iteration
![](1st_iteration.gif)

## Agent that has started to learn how to sail round the "course"
![](better_one_2.gif)

Unforunately the limit for GIF files is small on GitHub so the videos are short, but hopefully it indicates that the agent began to train! It managed to complete 4 laps of the course although not at a very efficient standard. Currently a human player of the game is much better than the AI but with further development this should not be the case. 

## Reward Timeline
Initailly the rewards just came from going through the reward gates and negative reward for hitting the wall. Since the gaps between reward gates were very large this meant that it struggled to train as it couldn't "find" the gate.

To try and counteract this the agent gained one reward every step it took closer to the gates and minus one reward when it went away. Unfortuantely this caused the agent to converge to crashing into the wall ASAP as this would cause a greater total reward, due to this the reward values were balanced better by increasing the negative aspect of hitting a wall.

Although both these reward criteria enabled better training there was no time dependancy so the agent was in no rush. To counteract this a dynamic reward total for going through the gate was created, the less number of steps taken between gates the more reward points were recieved. To addapt the distance from gate reward the closer to the gate the agent was the more reward points it would get, and the further away the greater the negative reward.

These changes greatly improved the models training.

## Next Steps:
* Further work developing the NN to enable better learning.
* Rather than 3 set actions have a vector space to decide how far to turn the boat left or right
* Create boat inertia so that the boat doesn't just stop when it is turning
* Inverse RL - it would be an interesting idea to see what reward criteria this method creates and whether it would improve model performance
* Dynamic wind direction and speed
* Add the ability to foil, will add further complexity as foiling is dependant on boat velocity being great enough

## Libraries:
* Pygame
* Tensorflow
* Keras
