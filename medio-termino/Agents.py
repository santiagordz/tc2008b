import math
from mesa import Agent


class Box(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class StackBox(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.isFull = False
        self.count = 0

    def addBox(self):
        self.count += 1
        if self.count == 5:
            self.isFull = True


# Robots
class Robot(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.hasBox = False
        self.destination = None

    def step(self):
        # Si no tenemos caja y la celda actual no es caja
        if not self.hasBox and not self.orderBox():
            self.moveRandom()
        # Si recogemos una caja
        elif self.orderBox():
            pass
        # Si tenemos caja
        elif self.hasBox:
            self.move2stack()

    def moveRandom(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        new_position = self.random.choice(possible_steps)

        # Checar que la nueva posición no esté ocupada
        cellmates = self.model.grid.get_cell_list_contents([new_position])
        if len(cellmates) > 0:
            for c in cellmates:
                if isinstance(c, Robot):
                    self.model.grid.move_agent(self, self.pos)
                    return

        self.model.grid.move_agent(self, new_position)

    def move2stack(self):
        # Si el robot esta cargando una caja, encontrar e ir a la pila más cercana
        nearest_stack = None
        min_distance = float("inf")
        for agent in self.model.schedule.agents:
            if isinstance(agent, StackBox) and not agent.isFull:
                distance = math.dist(self.pos, agent.pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_stack = agent

        # If a stack is found, set it as the destination
        if nearest_stack is not None:
            self.destination = nearest_stack.pos
            self.move_towards_destination()

    def move_towards_destination(self):
        if self.destination is None:
            return
        x, y = self.pos
        dest_x, dest_y = self.destination

        dx = 1 if dest_x > x else -1 if dest_x < x else 0
        dy = 1 if dest_y > y else -1 if dest_y < y else 0

        new_x = x + dx
        new_y = y + dy

        self.pos = (new_x, new_y)

    def orderBox(self):
        # Si la celda actual tiene una caja, recogerla
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 0:
            for c in cellmates:
                if isinstance(c, Box):
                    self.model.grid.remove_agent(c)
                    self.hasBox = True
                    return True
        return False

    def updateStackBox(self, x, y):
        for agent in self.model.schedule.agents:
            if isinstance(agent, StackBox) and agent.pos == (x, y) and not agent.isFull:
                agent.count += 1
                if agent.count == 5:
                    agent.isFull = True

    def checkStackBoxState(self):
        for agent in self.model.schedule.agents:
            if isinstance(agent, StackBox) and agent.count >= 5:
                agent.isFull = True
