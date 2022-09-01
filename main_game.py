import argparse

import gym
import pygame
from math import sin, cos, radians

from gym import spaces
from pygame.math import Vector2
import os, random
from PIL import Image

from max_speed import max_speed

# from player_movement import moves

from functions import get_angle, linesCollided, getCollisionPoint, dist, \
    get_direction, angle_to_wind, vel1, vel2, vel3, vel4

from display_size import displayHeight, displayWidth

import numpy as np


class Wall:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = displayHeight - y1
        self.x2 = x2
        self.y2 = displayHeight - y2

    def draw(self, surface):
        pygame.draw.line(surface, (50, 120, 100), (self.x1, self.y1), (self.x2, self.y2), width=10)

    """
    returns true if the car object has hit this wall
    """

    def hitCar(self, car):
        cw = car.width
        # since the car sprite isn't perfectly square the hitbox is a little smaller than the width of the car
        ch = car.height - 4
        rightVector = Vector2(car.direction)
        upVector = Vector2(car.direction).rotate(-90)
        carCorners = []
        cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        carPos = car.position
        for i in range(4):
            carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
                              (upVector * ch / 2 * cornerMultipliers[i][1]))

        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
                             carCorners[j].y):
                return True
        return False


class RewardGate:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.active = True

        self.center = Vector2((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def draw(self, surface):
        if self.active:
            pygame.draw.line(surface, (255, 0, 255), (self.x1, self.y1), (self.x2, self.y2), width=2)

    """
    returns true if the car object has hit this wall
    """

    def hitCar(self, car):
        if not self.active:
            return False

        cw = car.width
        # since the car sprite isn't perfectly square the hitbox is a little smaller than the width of the car
        ch = car.height - 4
        rightVector = Vector2(car.direction)
        upVector = Vector2(car.direction).rotate(-90)
        carCorners = []
        cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        carPos = car.position
        for i in range(4):
            carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
                              (upVector * ch / 2 * cornerMultipliers[i][1]))

        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
                             carCorners[j].y):
                return True
        return False


class Boat:
    def __init__(self, x, y, walls, rewardGates, args, width=10, height=15):
        self.team_racing = args.team_racing
        if self.team_racing:
            self.x = 320
            self.y = 620
            self.angle = 30
        else:
            self.x = x
            self.y = y
            self.angle = -30
        self.position = Vector2(self.x, self.y)
        self.speed = 0

        self.max_ep_len = args.max_ep_len

        self.direction = get_direction(self.angle)

        self.dead = False
        self.width = width
        self.height = height
        self.lineCollisionPoints = []
        self.collisionLineDistances = []
        self.vectorLength = displayWidth  # length of vision vectors

        self.turningLeft = False
        self.turningRight = False
        self.walls = walls
        self.rewardGates = rewardGates
        self.rewardNo = 0

        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - self.position

        self.reward = 0

        self.score = 0
        self.lifespan = 0

        self.steps_between_gate = 0
        self.dist_between_gate = self.directionToRewardGate.length()

        self.image = pygame.image.load("./boaty_boat.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (width, height))
        self.image_orig = self.image.copy()

        self.max_speed = max_speed

        self.max_length_on_screen = (displayHeight ** 2 + displayWidth ** 2) ** 0.5  # distance from corner to corner

        self.last_distance_reward_gate = self.max_length_on_screen

        self.continuous_action_space = args.has_continuous_action_space
        self.angle_adjust = 0
        self.angle_multiplier = args.max_steer


    def reset(self):
        self.position = Vector2(self.x, self.y)
        self.speed = 0
        self.angle = -30

        self.direction = get_direction(self.angle)

        self.dead = False
        self.lineCollisionPoints = []
        self.collisionLineDistances = []

        self.turningLeft = False
        self.turningRight = False

        self.rewardNo = 0
        self.reward = 0

        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - self.position

        self.steps_between_gate = 0
        self.dist_between_gate = self.directionToRewardGate.length()

        self.lifespan = 0
        self.score = 0
        for g in self.rewardGates:
            g.active = True

        self.last_distance_reward_gate = self.max_length_on_screen

        self.angle_adjust = 0

    def checkRewardGates(self):
        self.reward = 0
        if self.rewardGates[self.rewardNo].hitCar(self):
            self.rewardGates[self.rewardNo].active = False
            self.rewardNo += 1
            self.score += 1
            self.reward = self.max_ep_len * ((self.max_ep_len - self.steps_between_gate) / self.max_ep_len)  # adds time pressure
            self.steps_between_gate = 0
            if self.rewardNo == len(self.rewardGates):
                self.dist_between_gate = 100
            else:
                self.directionToRewardGate = self.rewardGates[self.rewardNo].center - self.position
                self.dist_between_gate = self.directionToRewardGate.length()
            # state = self.getState()
            # print(state[-3])
            if self.rewardNo == len(self.rewardGates):
                self.rewardNo = 0
                for g in self.rewardGates:
                    g.active = True

        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - self.position

        if self.last_distance_reward_gate > self.directionToRewardGate.length():
            self.reward += ((self.dist_between_gate - self.directionToRewardGate.length()) / self.dist_between_gate)
        elif self.last_distance_reward_gate < self.directionToRewardGate.length():
            self.reward -= 1 - ((self.dist_between_gate - self.directionToRewardGate.length()) / self.dist_between_gate)

        # print('{} last'.format(self.last_distance_reward_gate))
        # print('{} now'.format(self.directionToRewardGate.length()))

        self.last_distance_reward_gate = self.directionToRewardGate.length()

        return

    def hitAWall(self):
        """
        checks every wall and if the car has hit a wall returns true
        """
        for wall in self.walls:
            if wall.hitCar(self):
                return True

        return False

    def getState(self):
        self.setVisionVectors()
        normalizedVisionVectors = [1 - (max(1.0, line) / self.vectorLength) for line in self.collisionLineDistances]
        normalizedVisionVectors = [value + 1 if value == 0 else value for value in normalizedVisionVectors]
        normalizedVisionVectors = [values if values >= 0.97 else 0 for values in normalizedVisionVectors]

        normalizedAngleOfNextGate = (get_angle(self.direction) - get_angle(self.directionToRewardGate)) % 360
        if normalizedAngleOfNextGate > 180:
            normalizedAngleOfNextGate = -1 * (360 - normalizedAngleOfNextGate)

        normalizedAngleOfNextGate /= 180

        normalizedDirecToRewardGate_x = self.directionToRewardGate[0] / displayWidth
        normalizedDirecToRewardGate_y = self.directionToRewardGate[1] / displayHeight

        normalizedDistToRewardGate = self.directionToRewardGate.length() / self.max_length_on_screen

        normalizedAngleToWind = angle_to_wind(self.angle) / np.pi

        normalizedSpeed = self.speed / self.max_speed

        normalizedState = [*normalizedVisionVectors,  # 0 to 1
                           normalizedAngleOfNextGate,
                           # normalizedDirecToRewardGate_x, #-1 to 1
                           # normalizedDirecToRewardGate_y, #-1 to 1
                           normalizedDistToRewardGate,  # 0 to 1
                           normalizedAngleToWind,  # 0 to 1
                           normalizedSpeed]  # 0 to 1
        # print(normalizedState)
        return np.array(normalizedState)

    def setVisionVectors(self):
        """
        by creating lines in many directions from the car and getting the closest collision point of that line
        we create  "vision vectors" which will allow the car to 'see'
        kinda like a sonar system
        """
        h = self.height  # - 4
        w = self.width
        self.collisionLineDistances = []
        self.lineCollisionPoints = []
        self.setVisionVector(w / 2, 0, 0)

        self.setVisionVector(w / 2, -h / 2, -180 / 16)
        self.setVisionVector(w / 2, -h / 2, -180 / 4)
        # self.setVisionVector(w / 2, -h / 2, -4 * 180 / 8)

        self.setVisionVector(w / 2, h / 2, 180 / 16)
        self.setVisionVector(w / 2, h / 2, 180 / 4)
        # self.setVisionVector(w / 2, h / 2, 4 * 180 / 8)

        # self.setVisionVector(-w / 2, -h / 2, -6 * 180 / 8)
        # self.setVisionVector(-w / 2, h / 2, 6 * 180 / 8)
        # self.setVisionVector(-w / 2, 0, 180)

    def getPositionOnCarRelativeToCenter(self, right, up):
        rightVector = Vector2(self.direction)
        rightVector.normalize()
        upVector = self.direction.rotate(90)
        upVector.normalize()

        return self.position + ((rightVector * right) + (upVector * up))

    def getCollisionPointOfClosestWall(self, x1, y1, x2, y2):
        """
        returns the point of collision of a line (x1,y1,x2,y2) with the walls,
        if multiple walls are hit it returns the closest collision point
        """
        minDist = 2 * displayWidth
        closestCollisionPoint = Vector2(0, 0)
        for wall in self.walls:
            collisionPoint = getCollisionPoint(x1, y1, x2, y2, wall.x1, wall.y1, wall.x2, wall.y2)
            if collisionPoint is None:
                continue
            if dist(x1, y1, collisionPoint.x, collisionPoint.y) < minDist:
                minDist = dist(x1, y1, collisionPoint.x, collisionPoint.y)
                closestCollisionPoint = Vector2(collisionPoint)
        return closestCollisionPoint

    def setVisionVector(self, startX, startY, gamma):
        """
        calculates and stores the distance to the nearest wall given a vector
        """
        collisionVectorDirection = self.direction.rotate(gamma)
        collisionVectorDirection = collisionVectorDirection.normalize() * self.vectorLength
        startingPoint = self.getPositionOnCarRelativeToCenter(startX, startY)
        collisionPoint = self.getCollisionPointOfClosestWall(startingPoint.x, startingPoint.y,
                                                             startingPoint.x + collisionVectorDirection.x,
                                                             startingPoint.y + collisionVectorDirection.y)
        if collisionPoint.x == 0 and collisionPoint.y == 0:
            self.collisionLineDistances.append(self.vectorLength)
        else:
            self.collisionLineDistances.append(
                dist(startingPoint.x, startingPoint.y, collisionPoint.x, collisionPoint.y))
        self.lineCollisionPoints.append(collisionPoint)

    def showCollisionVectors(self, surface):
        """
        shows dots where the collision vectors detect a wall
        """
        for point in self.lineCollisionPoints:
            if point != [0, 0]:
                pygame.draw.line(surface, (131, 139, 139), (self.position.x, self.position.y), (point.x, point.y,), 1)
                pygame.draw.circle(surface, (0, 0, 0), (point.x, point.y), 5)

    def updateWithAction(self, actionNo, value_2):
        if self.continuous_action_space:
            self.angle_adjust = actionNo * self.angle_multiplier
        else:
            self.turningLeft = False
            self.turningRight = False

            if actionNo == 2:
                self.turningLeft = True
            elif actionNo == 1:
                self.turningRight = True
            elif actionNo == 0:
                pass

        totalReward = 0

        for i in range(1):
            if not self.dead:
                self.lifespan += 1
                self.steps_between_gate += 1
                self.move()

                if self.hitAWall():
                    self.dead = True
                    totalReward -= 100000
                    # return

                if self.angle < 30 and self.angle > -30:
                    totalReward -= 100

                # totalReward += self.reward
                # add reward for making it not go in the no go zone

                if self.score == 10:  # finishes game after one lap
                    self.dead = True
                    totalReward += 10000000 / value_2  # the shorter time step run the higher reward

                self.checkRewardGates()
                totalReward += self.reward

        self.setVisionVectors()

        self.reward = totalReward

        return self.getState(), self.reward, self.dead, {}

    def move(self):

        self.speed = self.rew(angle_to_wind(self.angle)) * 4

        self.position.x = self.position.x - (self.speed * sin(radians(self.angle)))
        self.position.y = self.position.y - (self.speed * cos(radians(self.angle)))

        self.direction = get_direction(self.angle)

        if self.continuous_action_space:
            self.angle += self.angle_adjust

        else:
            if self.turningRight:
                self.angle -= 4
            elif self.turningLeft:
                self.angle += 4

    def rew(self, theta, theta_0=0, theta_dead=np.pi / 12):
        if angle_to_wind(self.angle) <= 7 * np.pi / 36:
            return vel1(theta, theta_0, theta_dead) * np.cos(theta)
        elif angle_to_wind(self.angle) > 7 * np.pi / 36 and angle_to_wind(self.angle) <= 5 * np.pi / 8:
            return vel2(theta)
        elif angle_to_wind(self.angle) > 5 * np.pi / 8 and angle_to_wind(self.angle) <= 3 * np.pi / 4:
            return vel3(theta)
        elif angle_to_wind(self.angle) > 3 * np.pi / 4 and angle_to_wind(self.angle) <= np.pi:
            return vel4(theta)


class Game(gym.Env):
    def __init__(self, player, args):
        pygame.init()
        pygame.display.set_caption("RL Boat")
        width = displayWidth
        height = displayHeight
        self.screen = pygame.display.set_mode((width, height))
        self.myfont = pygame.font.SysFont("monospace", 16)

        self.team_racing = args.team_racing

        if player == 'Human':
            self.human = True
            self.clock = pygame.time.Clock()
            self.ticks = 60
        else:
            self.human = False

        self.exit = False

        self.walls = []
        self.gates = []
        self.set_walls()
        self.set_gates()
        self.boat = Boat(776, 640, self.walls, self.gates, args)

        if self.boat.continuous_action_space:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32).shape[0]
        else:
            self.action_space = 3
        self.state_space = 9

        self.take_pics = False

    def set_walls(self):
        self.walls.append(Wall(0, -100, 1, displayHeight + 100))
        self.walls.append(Wall(-100, 0, displayWidth + 100, 1))
        self.walls.append(Wall(displayWidth, -100, displayWidth - 1, displayHeight + 100))
        self.walls.append(Wall(-100, displayHeight, displayWidth + 100, displayHeight - 1))

    def set_gates(self):
        if self.team_racing:
            self.gates.append(RewardGate(260, 600, 340, 600))
            self.gates.append(RewardGate(240, 100, 290, 100))
            self.gates.append(RewardGate(300, 90, 300, 40))
            self.gates.append(RewardGate(650, 90, 650, 40))
            self.gates.append(RewardGate(660, 100, 710, 100))
            self.gates.append(RewardGate(590, 600, 640, 600))
            self.gates.append(RewardGate(650, 610, 650, 660))
            self.gates.append(RewardGate(1000, 610, 1000, 660))
            self.gates.append(RewardGate(1010, 600, 1060, 600))
            self.gates.append(RewardGate(960, 100, 1040, 100))

        else:
            self.gates.append(RewardGate(750, 600, 850, 600))
            self.gates.append(RewardGate(810, 100, 860, 100))
            self.gates.append(RewardGate(800, 90, 800, 40))
            self.gates.append(RewardGate(350, 140, 350, 90))
            self.gates.append(RewardGate(290, 150, 340, 150))
            self.gates.append(RewardGate(290, 600, 340, 600))
            self.gates.append(RewardGate(350, 610, 350, 660))
            self.gates.append(RewardGate(699, 644, 699, 555))

    def get_state(self):
        state = self.boat.getState()
        # return np.reshape(state, (1, self.state_space))
        return state
        pass

    def step(self, action, value, value_2):
        s, r, d, _ = self.boat.updateWithAction(action, value_2)
        self.render()
        if self.take_pics:
            self.save_pics(value)
        return s, r, d, _

    def reset(self):
        self.boat.reset()
        self.take_pics = False

        return self.get_state()

    def render(self):
        state = self.get_state()
        state = state[-4]
        state *= 180
        self.screen.fill((127, 225, 212))

        for i in self.walls:
            i.draw(self.screen)
        for j in self.gates:
            j.draw(self.screen)

        self.boat.showCollisionVectors(self.screen)
        rotated = pygame.transform.rotate(self.boat.image_orig, self.boat.angle)
        rect = rotated.get_rect()
        self.screen.blit(rotated, self.boat.position - (rect.width / 2, rect.height / 2))

        if self.team_racing:
            pygame.draw.circle(self.screen, (0, 0, 0), (250, 600), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (350, 600), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (300, 100), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (650, 100), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (650, 600), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (1000, 600), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (950, 100), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (1050, 100), 4)

        else:
            pygame.draw.circle(self.screen, (0, 0, 0), (800, 100), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (350, 150), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (350, 600), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (700, 650), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (700, 550), 4)

        speedtext = self.myfont.render("Speed = " + str(int(self.boat.speed * 5)), 1, (0, 0, 0))
        self.screen.blit(speedtext, (7, 10))
        scoretext = self.myfont.render("Score = " + str(int(self.boat.score)), 1, (0, 0, 0))
        self.screen.blit(scoretext, (122, 10))
        rewardtext = self.myfont.render("Direc to Reward = " + str(self.boat.directionToRewardGate), 1, (0, 0, 0))
        self.screen.blit(rewardtext, (235, 10))
        gatetext = self.myfont.render("Angle To RwdGate = " + str(state), 1, (0, 0, 0))
        self.screen.blit(gatetext, (640, 10))

        pygame.display.update()

        if self.human:
            self.clock.tick(self.ticks)
        else:
            pass

        pygame.event.pump()

    def save_pics(self, value):
        self.display_surface = pygame.display.get_surface()

        self.image3d = np.ndarray(
            (displayWidth, displayHeight, 3), np.uint8)

        pygame.pixelcopy.surface_to_array(
            self.image3d, self.display_surface)
        self.image3dT = np.transpose(self.image3d, axes=[1, 0, 2])
        im = Image.fromarray(self.image3dT)  # monochromatic image
        imrgb = im.convert('RGB')  # color image

        filename = ''.join(['Episode_',
                            str(value),
                            '-frame-',
                            str(self.boat.lifespan).zfill(5),
                            '.jpg'])
        foldername = ''.join(['./exported_frames/Episode_', str(value)])
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        filenamepath = os.path.join(foldername, filename)
        imrgb.save(filenamepath)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='Test_1')
    parser.add_argument('--has_continuous_action_space', default=True)
    parser.add_argument('--max_steer', default=10)
    parser.add_argument('--max_ep_len', default=10000)
    parser.add_argument('--team_racing', default=True)
    # parser.add_argument('--team_racing', default=False)
    args = parser.parse_args()
    env = Game('Human', args)
    time_steps = 10000  # 800
    for _ in range(time_steps):
        s, r, d, _ = env.step(random.randint(0, 2), 'e')
        # print(f'state: {s}')
        # print(f'reward: {r}')
        env.render()
    # env.close()


