import pygame
import random
import math
import matplotlib.pyplot as plt

# ---------------- Simulation Parameters ----------------
WIDTH, HEIGHT = 800, 600

# Agent parameters
NUM_INITIAL_AGENTS = 30
INITIAL_ENERGY = 100
REPRODUCTION_THRESHOLD = 150
REPRODUCTION_COST = 50
MUTATION_STD = 0.2  # standard deviation for mutating sensitivity and speed
BASE_METABOLISM = 0.1
SPEED_COST_FACTOR = 0.02  # extra energy cost per unit speed

# Food parameters
FOOD_ENERGY = 20
FOOD_SPAWN_PROB = 0.05  # probability of spawning food each frame
MAX_FOOD = 70

# Sensing and communication ranges
AGENT_FOOD_RANGE = 80  # range within which an agent can detect food
AGENT_DANGER_RANGE = 120  # range within which an agent can detect the predator
COMMUNICATION_RANGE = 200  # range for receiving neighbor signals
AGENT_WANDER_CHANGE = 0.1  # random change in direction when nothing is sensed

# Predator parameters
PREDATOR_SPEED = 2.55
PREDATOR_DETECTION_RANGE = 200  # predator looks for agents within this range
PREDATOR_CATCH_RANGE = 15  # distance for predator to "catch" an agent

# Colors (RGB)
BACKGROUND_COLOR = (30, 30, 30)
AGENT_COLOR = (0, 255, 0)
PREDATOR_COLOR = (255, 0, 0)
FOOD_COLOR = (255, 255, 0)


# ------------------ Agent Class ------------------
class Agent:
    def __init__(self, x, y, speed, sensitivity, energy):
        self.x = x
        self.y = y
        self.speed = speed
        self.sensitivity = sensitivity  # how strongly the agent weighs neighbor signals
        self.energy = energy
        self.direction = random.uniform(0, 2 * math.pi)
        self.signal = None  # (dx, dy, type) emitted each frame

    def detect_environment(self, food_list, predator):
        """
        Detect food and danger. If the predator is close, override food with a danger avoidance direction.
        Returns:
          danger_detected (bool),
          food_detected (Food or None),
          base_direction (radians)
        """
        danger_detected = False
        base_direction = None

        # Check for predator danger
        d_pred = math.hypot(predator.x - self.x, predator.y - self.y)
        if d_pred < AGENT_DANGER_RANGE:
            danger_detected = True
            # Set base direction away from predator
            base_direction = math.atan2(self.y - predator.y, self.x - predator.x)

        # If no danger, check for food
        food_detected = None
        if not danger_detected:
            closest_food = None
            min_food_dist = float('inf')
            for food in food_list:
                d = math.hypot(food.x - self.x, food.y - self.y)
                if d < AGENT_FOOD_RANGE and d < min_food_dist:
                    min_food_dist = d
                    closest_food = food
            if closest_food:
                food_detected = closest_food
                base_direction = math.atan2(closest_food.y - self.y, closest_food.x - self.x)

        # If nothing is detected, continue wandering
        if base_direction is None:
            base_direction = self.direction + random.uniform(-AGENT_WANDER_CHANGE, AGENT_WANDER_CHANGE)

        return danger_detected, food_detected, base_direction

    def emit_signal(self, danger_detected, food_detected, base_direction):
        """
        Emit a signal based on what is detected.
        Danger signals (when predator is near) are more urgent.
        """
        if danger_detected:
            # Danger signal: vector pointing away from the predator
            self.signal = (math.cos(base_direction), math.sin(base_direction), "danger")
        elif food_detected:
            self.signal = (math.cos(base_direction), math.sin(base_direction), "food")
        else:
            self.signal = None

    def gather_signals(self, agents):
        """
        Gather and average signals from neighboring agents within COMMUNICATION_RANGE.
        Danger signals are weighted more heavily.
        """
        total_vec = [0, 0]
        count = 0
        for agent in agents:
            if agent is self:
                continue
            d = math.hypot(agent.x - self.x, agent.y - self.y)
            if d < COMMUNICATION_RANGE and agent.signal is not None:
                vec = [agent.signal[0], agent.signal[1]]
                if agent.signal[2] == "danger":
                    # Danger signals have double weight
                    vec[0] *= 2
                    vec[1] *= 2
                total_vec[0] += vec[0]
                total_vec[1] += vec[1]
                count += 1
        if count > 0:
            total_vec[0] /= count
            total_vec[1] /= count
            return total_vec
        return None

    def update(self, food_list, predator, agents):
        """
        Update agent behavior:
         - Detect environment to set a base direction.
         - Emit a signal (food or danger) accordingly.
         - Gather neighbor signals and combine with the base direction.
         - Update position, wrap around boundaries, and expend energy.
        """
        danger_detected, food_detected, base_direction = self.detect_environment(food_list, predator)
        self.emit_signal(danger_detected, food_detected, base_direction)
        neighbor_signal = self.gather_signals(agents)

        # Combine base direction with weighted neighbor signals
        dx = math.cos(base_direction)
        dy = math.sin(base_direction)
        if neighbor_signal is not None:
            dx += self.sensitivity * neighbor_signal[0]
            dy += self.sensitivity * neighbor_signal[1]
        new_direction = math.atan2(dy, dx)
        self.direction = new_direction

        # Update position
        self.x += math.cos(self.direction) * self.speed
        self.y += math.sin(self.direction) * self.speed
        self.x %= WIDTH
        self.y %= HEIGHT

        # Energy cost for moving and metabolism
        self.energy -= BASE_METABOLISM + self.speed * SPEED_COST_FACTOR

    def eat(self, food):
        """Increase energy when food is consumed."""
        self.energy += FOOD_ENERGY

    def reproduce(self):
        """
        Reproduce if energy is high enough.
        Offspring inherit speed and sensitivity with small mutations.
        """
        new_sensitivity = self.sensitivity + random.gauss(0, MUTATION_STD)
        new_sensitivity = max(0, new_sensitivity)  # Ensure sensitivity is non-negative
        new_speed = self.speed + random.gauss(0, MUTATION_STD)
        new_speed = max(0.5, new_speed)
        offspring = Agent(self.x, self.y, new_speed, new_sensitivity, INITIAL_ENERGY)
        self.energy -= REPRODUCTION_COST
        return offspring

    def draw(self, surface):
        """Draw the agent and a small line indicating its current direction."""
        size = 4
        pygame.draw.circle(surface, AGENT_COLOR, (int(self.x), int(self.y)), size)
        end_x = self.x + math.cos(self.direction) * 10
        end_y = self.y + math.sin(self.direction) * 10
        pygame.draw.line(surface, AGENT_COLOR, (self.x, self.y), (end_x, end_y), 1)


# ------------------ Predator Class ------------------
class Predator:
    def __init__(self, x, y, speed=PREDATOR_SPEED):
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = random.uniform(0, 2 * math.pi)

    def update(self, agents):
        """
        Predator looks for the nearest agent within its detection range.
        If one is found, it moves toward that agent.
        Otherwise, it wanders randomly.
        """
        target = None
        min_dist = float('inf')
        for agent in agents:
            d = math.hypot(agent.x - self.x, agent.y - self.y)
            if d < PREDATOR_DETECTION_RANGE and d < min_dist:
                min_dist = d
                target = agent
        if target:
            self.direction = math.atan2(target.y - self.y, target.x - self.x)
        else:
            self.direction += random.uniform(-0.1, 0.1)
        self.x += math.cos(self.direction) * self.speed
        self.y += math.sin(self.direction) * self.speed
        self.x %= WIDTH
        self.y %= HEIGHT

    def draw(self, surface):
        """Draw the predator as a larger circle."""
        size = 8
        pygame.draw.circle(surface, PREDATOR_COLOR, (int(self.x), int(self.y)), size)


# ------------------ Food Class ------------------
class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, surface):
        pygame.draw.circle(surface, FOOD_COLOR, (int(self.x), int(self.y)), 3)


# ------------------ Main Simulation ------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Co-evolution of Communication & Collective Decision-Making")
    clock = pygame.time.Clock()

    # Initialize agents with random positions, speeds, and sensitivities
    agents = []
    for _ in range(NUM_INITIAL_AGENTS):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        speed = random.uniform(1, 3)
        sensitivity = random.uniform(0, 1)
        agents.append(Agent(x, y, speed, sensitivity, INITIAL_ENERGY))

    # Initialize a single predator
    predator = Predator(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))

    # Food storage
    food_list = []

    # Data tracking for plots: average sensitivity and population over time
    time_points = []
    avg_sensitivities = []
    populations = []
    frame_count = 0
    stat_interval = 30  # record statistics every 60 frames (about one second at 60 FPS)

    running = True
    while running:
        clock.tick(60)  # 60 FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Spawn food with a probability per frame (and cap the total number)
        if random.random() < FOOD_SPAWN_PROB and len(food_list) < MAX_FOOD:
            food_list.append(Food(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)))

        new_agents = []
        # Update each agent
        for agent in agents:
            agent.update(food_list, predator, agents)

            # Check if the agent eats food (within 10 pixels)
            for food in food_list:
                if math.hypot(food.x - agent.x, food.y - agent.y) < 10:
                    agent.eat(food)
                    if food in food_list:
                        food_list.remove(food)
                    break

            # If the predator is too close, the agent is "caught"
            if math.hypot(predator.x - agent.x, predator.y - agent.y) < PREDATOR_CATCH_RANGE:
                agent.energy = 0

            # Reproduction if energy is high enough
            if agent.energy > REPRODUCTION_THRESHOLD:
                offspring = agent.reproduce()
                new_agents.append(offspring)

            # Only keep agents with positive energy
            if agent.energy > 0:
                new_agents.append(agent)
        agents = new_agents

        # Update the predator’s movement
        predator.update(agents)

        # Record simulation statistics periodically
        if frame_count % stat_interval == 0:
            time_points.append(frame_count // 60)
            if agents:
                avg_sensitivity = sum(a.sensitivity for a in agents) / len(agents)
            else:
                avg_sensitivity = 0
            avg_sensitivities.append(avg_sensitivity)
            populations.append(len(agents))

        # Draw the simulation scene
        screen.fill(BACKGROUND_COLOR)
        for food in food_list:
            food.draw(screen)
        for agent in agents:
            agent.draw(screen)
        predator.draw(screen)
        pygame.display.flip()

        frame_count += 1
        # End simulation if all agents have perished or after a fixed number of frames
        if len(agents) == 0 or frame_count > 6000:
            running = False

    pygame.quit()

    # ------------------ Plotting Statistics ------------------
    plt.figure(figsize=(12, 6))

    # Plot evolution of average communication sensitivity
    plt.subplot(1, 2, 1)
    plt.plot(time_points, avg_sensitivities, label="Avg Communication Sensitivity")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Average Sensitivity")
    plt.title("Evolution of Communication Sensitivity")
    plt.legend()

    # Plot population dynamics over time
    plt.subplot(1, 2, 2)
    plt.plot(time_points, populations, label="Population", color='orange')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Population")
    plt.title("Population Dynamics")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
