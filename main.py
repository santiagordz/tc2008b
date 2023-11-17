from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams["animation.embed_limit"] = 2**128

random.seed(67890)


class Box(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 1


class Robot(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 2
        self.box_carried = False
        self.box_carried_id = None
        self.has_tower = False
        self.tower_coord = None

    def moveRandom(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def move2empty(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        new_position = random.choice(possible_steps)
        while self.model.grid[new_position[0]][new_position[1]] != None:
            new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def look4tower(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )
        has_box = False
        for neighbor in neighbors:
            if isinstance(neighbor, Box):
                has_box = True
        if has_box:
            self.markTower()
        else:
            self.moveRandom()

    def markTower(self):
        possibles = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )
        for possible in possibles:
            if possible in self.model.towers:
                possibles.remove(possible)

        # Get Neighbors Positions
        positions = set()
        for possible in possibles:
            pos = possible.pos
            positions.add(pos)

        # Get the highest box count from neighbors
        count = 0
        tower_pos = None
        for position in positions:
            box_count = self.model.boxes[position[0]][position[1]]
            if box_count > count:
                count = box_count
                tower_pos = position
                # Set the robot tower coord
                self.tower_coord = tower_pos

        # Mark the tower
        self.has_tower = True
        self.model.towers.append(tower_pos)

    def look4Box(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )
        box_neighbor = []

        for neighbor in neighbors:
            if isinstance(neighbor, Box):
                box_neighbor.append(neighbor)

        box_coords = []
        for box in box_neighbor:
            box_coords.append(box.pos)

        box_coords = list(set(box_coords))
        for tower in self.model.towers:
            if tower in box_coords:
                box_coords.remove(tower)

        # Delete box from neighbor if neighbor coord is in box_coords
        for box in box_neighbor:
            if box.pos in box_coords:
                box_neighbor.remove(box)

        # Pick a box
        if len(box_coords) > 0:
            box = random.choice(box_neighbor)
            self.model.pick_box(box.pos)
            self.box_carried_id = box.unique_id
            self.box_carried = True

        else:
            self.moveRandom()

    def checkTower(self):
        if self.model.boxes[self.pos[0]][self.pos[1]] == 5:
            self.has_tower = False
            self.tower_coord = None

    def move2tower(self):
        x1, y1 = self.pos
        x2, y2 = self.tower_coord

        if x1 < x2:
            x1 += 1
            self.model.grid.move_agent(self, (x1, y1))
            return
        elif x1 > x2:
            x1 += -1
            self.model.grid.move_agent(self, (x1, y1))
            return
        elif y1 < y2:
            y1 += 1
            self.model.grid.move_agent(self, (x1, y1))
            return
        elif y1 > y2:
            y1 += -1
            self.model.grid.move_agent(self, (x1, y1))
            return
        elif x1 == x2 and y1 == y2:
            self.model.drop_box(self.pos, self.box_carried_id)
            self.checkTower()
            self.box_carried = False
            self.move2empty()
            return

    def step(self):
        if not self.box_carried and not self.has_tower:
            self.look4tower()
        elif not self.box_carried and self.has_tower:
            self.look4Box()
        elif self.box_carried and self.has_tower:
            self.move2tower()


def get_grid(model):
    grid = np.zeros((model.width, model.height))
    for contents, (x, y) in model.grid.coord_iter():
        for content in contents:
            if isinstance(content, Box):
                grid[x][y] = model.boxes[x][y]
            elif isinstance(content, Robot):
                grid[x][y] = 10

    return grid


class WarehouseModel(Model):
    def __init__(self, width, height, num_agents, num_boxes):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.num_boxes = num_boxes

        self.boxes = np.zeros((width, height))
        self.towers = []

        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(model_reporters={"Grid": get_grid})

        # Place agents
        robotId = 0
        for _ in range(self.num_agents):
            a = Robot(robotId, self)
            self.schedule.add(a)
            pos = self.random_empty_cell()
            while self.is_box(pos):
                pos = self.random_empty_cell()
            self.grid.place_agent(a, pos)
            robotId += 1

        # Place boxes
        boxId = 10
        for _ in range(self.num_boxes):
            b = Box(boxId, self)
            self.schedule.add(b)
            pos = self.random_position()
            while self.init_full(pos):
                pos = self.random_position()
            self.grid.place_agent(b, pos)
            self.boxes[pos[0]][pos[1]] += 1
            boxId += 1

    def is_box(self, pos):
        return self.boxes[pos[0]][pos[1]] > 0

    def init_full(self, pos):
        return self.boxes[pos[0]][pos[1]] == 3

    def is_tower_full(self, pos):
        return self.boxes[pos[0]][pos[1]] == 5

    def pick_box(self, pos):
        self.boxes[pos[0]][pos[1]] -= 1
        self.grid.remove_agent(self.grid.get_cell_list_contents(pos)[0])

    def drop_box(self, pos, unique_id):
        b = Box(unique_id, self)
        self.boxes[pos[0]][pos[1]] += 1
        self.grid.place_agent(b, pos)

    def count_carried_boxes(self):
        return sum(agent.box_carried for agent in self.schedule.agents)

    def random_empty_cell(self):
        empty_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.grid.is_cell_empty((x, y))
        ]
        if not empty_cells:
            raise Exception("No empty cells available.")
        return random.choice(empty_cells)

    def random_position(self):
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        return (x, y)

    def is_simulation_done(self):
        return np.all(np.logical_or(self.boxes == 0, self.boxes == 5))

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


WIDTH = 20
HEIGHT = 20
NUM_AGENTS = 1
NUM_BOXES = 200

model = WarehouseModel(WIDTH, HEIGHT, NUM_AGENTS, NUM_BOXES)
print(model.towers())
# for i in range(100):
#     model.step()
