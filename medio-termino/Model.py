from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
import numpy as np
import random

random.seed(67890)

from Agents import *


def get_grid(model):
    grid = np.zeros((model.grid.width, model.grid.height))
    for content, (x, y) in model.grid.coord_iter():
        if content == None:
            grid[x][y] = 0
        elif isinstance(content, Robot):
            grid[x][y] = 5
        elif isinstance(content, Box):
            grid[x][y] = 1
        elif isinstance(content, StackBox):
            if content.count == 0:
                grid[x][y] = 2
            elif content.count == 1:
                grid[x][y] = 3
            elif content.count == 2:
                grid[x][y] = 4
            elif content.count == 3:
                grid[x][y] = 5
            elif content.count == 4:
                grid[x][y] = 6
            elif content.isFull:
                grid[x][y] = 7
    return grid


class OrderingRobotsModel(Model):
    def __init__(self):
        self.num_robots = 5
        self.num_stacks = random.randint(10, 50)
        self.width = 20
        self.height = 20

        self.grid = MultiGrid(self.width, self.height, False)
        self.schedule = RandomActivation(self)
        self.running = True
        self.datacollector = DataCollector(model_reporters={"Grid": get_grid})

        empty_cells = self.find_empty_cells()

        # Create Robots
        for i in range(self.num_robots):
            cell = random.choice(empty_cells)
            empty_cells.remove(cell)

            robot = Robot(i, self)
            self.schedule.add(robot)
            self.grid.place_agent(robot, cell)

        # Create Stacks
        for i in range(self.num_stacks):
            cell = random.choice(empty_cells)
            empty_cells.remove(cell)

            stack = StackBox(i, self)
            self.schedule.add(stack)
            self.grid.place_agent(stack, cell)

        # Distribute Boxes
        for i in range(200):
            available_stacks = [
                stack
                for stack in self.schedule.agents
                if isinstance(stack, StackBox) and not stack.isFull
            ]
            if random.choice([True, False]) and available_stacks:
                stack = random.choice(available_stacks)
                stack.addBox()

                box = Box(i, self)
                self.grid.place_agent(box, stack.pos)
                self.schedule.add(box)

            else:
                cell = random.choice(empty_cells)
                box = Box(i, self)
                self.grid.place_agent(box, cell)
                self.schedule.add(box)

    def find_empty_cells(self):
        found_empty_cells = []
        for contents, (x, y) in self.grid.coord_iter():
            if contents is None:
                found_empty_cells.append((x, y))
        return found_empty_cells

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
