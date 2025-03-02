import pygame
import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from tqdm import tqdm

# Configuration
ANIMATE = False       # Set to False to run simulation without animation.
NUM_STEPS = 10000     # Total simulation steps.
NUM_RUNS = 10         # Number of simulation runs to average over

# Constants
GRID_SIZE = 100       # Grid dimensions
VISION_RANGE = 5      # Distance agents can see resources
COMM_RANGE = 20       # Communication range (Euclidean)
CLUSTER_NUM = 7       # Number of resource clusters
CLUSTER_STRENGTH = 3  # Peak resource contribution per cluster
CLUSTER_SIGMA = 3     # Spread of resource clusters
RESOURCE_MAX = 4      # Maximum resource per cell
RESOURCE_REGEN = 0.01 # Regeneration rate factor (very low value)
COLLECTION_RATE = 1   # Max resources collected per step
ENERGY_COST = 0.2     # Energy cost per step
REPRODUCTION_THRESHOLD = 50  # Energy needed to reproduce
MUTATION_STD = 0.02   # Standard deviation for trait mutation
RESOURCE_THRESHOLD = 5# Minimum resource level to trigger signaling
SCREEN_SIZE = 1000    # Pygame window size (pixels)

# Dynamic simulation constants
EXPLORATION_RATE = 0.01  # Base probability an agent randomly explores (scaled by energy)
EVENT_INTERVAL = 100     # Steps between dynamic environmental events
EVENT_RADIUS = 20        # Radius of drought event (in grid cells)
EVENT_INTENSITY = 0.1    # Drought intensity factor (resources multiplied by this factor)

# Initialize Pygame only if animation is enabled.
if ANIMATE:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Evolutionary Communication Simulation")
    clock = pygame.time.Clock()

def toroidal_distance(a: int, b: int) -> int:
    """Calculate the minimum distance between two coordinates on a toroidal grid."""
    return min(abs(a - b), GRID_SIZE - abs(a - b))

class ResourceGrid:
    """Manages the resource distribution and dynamics on the grid."""
    def __init__(self) -> None:
        self.clusters = [{'x': random.uniform(0, GRID_SIZE), 'y': random.uniform(0, GRID_SIZE)}
                         for _ in range(CLUSTER_NUM)]
        self.resources = np.zeros((GRID_SIZE, GRID_SIZE))
        self.update_target_resources()

    def update_target_resources(self) -> None:
        x = np.arange(GRID_SIZE)
        y = np.arange(GRID_SIZE)
        X, Y = np.meshgrid(x, y, indexing='ij')
        self.target_resources = np.zeros((GRID_SIZE, GRID_SIZE))
        for cluster in self.clusters:
            dx = np.abs(X - cluster['x'])
            dx = np.minimum(dx, GRID_SIZE - dx)
            dy = np.abs(Y - cluster['y'])
            dy = np.minimum(dy, GRID_SIZE - dy)
            dist = np.sqrt(dx ** 2 + dy ** 2)
            self.target_resources += CLUSTER_STRENGTH * np.exp(-dist ** 2 / (2 * CLUSTER_SIGMA ** 2))

    def move_clusters(self) -> None:
        for cluster in self.clusters:
            if random.random() < 0.005:  # 0.5% chance for an abrupt change.
                cluster['x'] = random.uniform(0, GRID_SIZE)
                cluster['y'] = random.uniform(0, GRID_SIZE)
            else:
                cluster['x'] = (cluster['x'] + random.gauss(0, 0.1)) % GRID_SIZE
                cluster['y'] = (cluster['y'] + random.gauss(0, 0.1)) % GRID_SIZE
        self.update_target_resources()

    def update_resources(self) -> None:
        self.resources += RESOURCE_REGEN * (self.target_resources - self.resources)
        self.resources = np.clip(self.resources, 0, RESOURCE_MAX)

    def collect(self, x: int, y: int) -> float:
        collected = min(self.resources[x, y], COLLECTION_RATE)
        self.resources[x, y] -= collected
        return collected

@dataclass
class Agent:
    x: int
    y: int
    energy: float
    honesty: float  # Probability of sending true signal
    trust: float    # Probability of trusting received signal
    target: tuple = None

    def see_resources(self, grid: ResourceGrid) -> list:
        seen = []
        for dx in range(-VISION_RANGE, VISION_RANGE + 1):
            for dy in range(-VISION_RANGE, VISION_RANGE + 1):
                nx = (self.x + dx) % GRID_SIZE
                ny = (self.y + dy) % GRID_SIZE
                seen.append(((nx, ny), grid.resources[nx, ny]))
        return seen

    def find_best_seen(self, grid: ResourceGrid) -> tuple:
        seen = self.see_resources(grid)
        best_loc, best_val = max(seen, key=lambda item: item[1])
        return best_loc if best_val > grid.resources[self.x, self.y] else (self.x, self.y)

    def send_signal(self, grid: ResourceGrid) -> tuple:
        visible = [item for item in self.see_resources(grid) if item[1] > RESOURCE_THRESHOLD]
        if visible:
            best_loc, _ = max(visible, key=lambda item: item[1])
            if random.random() < self.honesty:
                return best_loc
            return (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        return None

    def move_towards(self, target: tuple) -> tuple:
        if target is None:
            return self.x, self.y
        dx = (target[0] - self.x + GRID_SIZE // 2) % GRID_SIZE - GRID_SIZE // 2
        dy = (target[1] - self.y + GRID_SIZE // 2) % GRID_SIZE - GRID_SIZE // 2
        if abs(dx) > abs(dy):
            step_x = int(np.sign(dx))
            step_y = 0
        else:
            step_x = 0
            step_y = int(np.sign(dy))
        return ((self.x + step_x) % GRID_SIZE, (self.y + step_y) % GRID_SIZE)

    def collect(self, grid: ResourceGrid) -> None:
        collected = grid.collect(self.x, self.y)
        self.energy += collected

    def reproduce(self) -> 'Agent':
        if self.energy > REPRODUCTION_THRESHOLD:
            offspring_energy = self.energy / 2
            self.energy = offspring_energy
            offspring_honesty = np.clip(self.honesty + random.gauss(0, MUTATION_STD), 0, 1)
            offspring_trust = np.clip(self.trust + random.gauss(0, MUTATION_STD), 0, 1)
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            new_x = (self.x + dx) % GRID_SIZE
            new_y = (self.y + dy) % GRID_SIZE
            return Agent(new_x, new_y, offspring_energy, offspring_honesty, offspring_trust)
        return None

class Simulation:
    def __init__(self, num_agents: int = 100) -> None:
        self.grid = ResourceGrid()
        self.agents = [Agent(random.randint(0, GRID_SIZE // 4),
                             random.randint(0, GRID_SIZE // 4),
                             50, random.random(), random.random())
                       for _ in range(num_agents)]
        self.time = 0
        self.data = {'time': [], 'avg_honesty': [], 'avg_trust': [], 'pop_size': []}

    def trigger_drought_event(self) -> None:
        center_x = random.randint(0, GRID_SIZE - 1)
        center_y = random.randint(0, GRID_SIZE - 1)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                dx = min(abs(x - center_x), GRID_SIZE - abs(x - center_x))
                dy = min(abs(y - center_y), GRID_SIZE - abs(y - center_y))
                if np.sqrt(dx ** 2 + dy ** 2) <= EVENT_RADIUS:
                    self.grid.resources[x, y] *= EVENT_INTENSITY

    def run_step(self) -> None:
        self.grid.move_clusters()
        self.grid.update_resources()
        if self.time != 0 and self.time % EVENT_INTERVAL == 0:
            self.trigger_drought_event()
        signals = []
        for agent in self.agents:
            sig = agent.send_signal(self.grid)
            if sig:
                signals.append((agent.x, agent.y, sig))
        for agent in self.agents:
            agent.collect(self.grid)
            for sx, sy, loc in signals:
                dx = toroidal_distance(sx, agent.x)
                dy = toroidal_distance(sy, agent.y)
                if np.hypot(dx, dy) < COMM_RANGE and random.random() < agent.trust:
                    agent.target = loc
            # Adjust exploration probability based on energy.
            exploration_prob = EXPLORATION_RATE * min(1, agent.energy / REPRODUCTION_THRESHOLD)
            if agent.target is None and random.random() < exploration_prob:
                agent.target = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if agent.target:
                new_pos = agent.move_towards(agent.target)
                if new_pos == agent.target:
                    agent.target = None
                agent.x, agent.y = new_pos
            else:
                best = agent.find_best_seen(self.grid)
                if best != (agent.x, agent.y):
                    agent.x, agent.y = agent.move_towards(best)
            agent.energy -= ENERGY_COST
        survivors = []
        new_agents = []
        for agent in self.agents:
            if agent.energy > 0:
                survivors.append(agent)
                offspring = agent.reproduce()
                if offspring:
                    new_agents.append(offspring)
        self.agents = survivors + new_agents
        if self.agents:
            avg_honesty = np.mean([a.honesty for a in self.agents])
            avg_trust = np.mean([a.trust for a in self.agents])
        else:
            avg_honesty = avg_trust = 0
        self.data['time'].append(self.time)
        self.data['avg_honesty'].append(avg_honesty)
        self.data['avg_trust'].append(avg_trust)
        self.data['pop_size'].append(len(self.agents))
        self.time += 1

    def draw(self) -> None:
        screen.fill((0, 0, 0))
        cell_size = SCREEN_SIZE // GRID_SIZE
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                intensity = int(255 * self.grid.resources[x, y] / RESOURCE_MAX)
                pygame.draw.rect(screen, (0, intensity, 0),
                                 (x * cell_size, y * cell_size, cell_size, cell_size))
        for agent in self.agents:
            color = (int(255 * agent.honesty), 0, int(255 * agent.trust))
            pygame.draw.circle(screen, color,
                               (agent.x * cell_size + cell_size // 2,
                                agent.y * cell_size + cell_size // 2), 2)
        pygame.display.flip()
        clock.tick(30)

    def run(self, steps: int, animate: bool = True, plot_results: bool = True) -> None:
        running = True
        for _ in tqdm(range(steps)):
            if animate:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                if not running:
                    break
                self.run_step()
                self.draw()
            else:
                self.run_step()
        if animate:
            pygame.quit()
        if plot_results:
            self.plot()

    def plot(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(self.data['time'], self.data['avg_honesty'], 'r-')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Average Honesty')
        axes[1].plot(self.data['time'], self.data['avg_trust'], 'b-')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Average Trust')
        axes[2].plot(self.data['time'], self.data['pop_size'], 'k-')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Population Size')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Run multiple simulations and aggregate data.
    all_time = []
    all_honesty = []
    all_trust = []
    all_pop = []
    for _ in tqdm(range(NUM_RUNS)):
        sim = Simulation(num_agents=100)
        sim.run(steps=NUM_STEPS, animate=ANIMATE, plot_results=False)
        all_time.append(np.array(sim.data['time']))
        all_honesty.append(np.array(sim.data['avg_honesty']))
        all_trust.append(np.array(sim.data['avg_trust']))
        all_pop.append(np.array(sim.data['pop_size']))

    # Assume time arrays are identical across runs.
    time_axis = all_time[0]
    mean_honesty = np.mean(all_honesty, axis=0)
    std_honesty = np.std(all_honesty, axis=0)
    mean_trust = np.mean(all_trust, axis=0)
    std_trust = np.std(all_trust, axis=0)
    mean_pop = np.mean(all_pop, axis=0)
    std_pop = np.std(all_pop, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(time_axis, mean_honesty, 'r-', label='Honesty')
    axes[0].fill_between(time_axis, mean_honesty - std_honesty, mean_honesty + std_honesty, color='r', alpha=0.3)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Average Honesty')
    axes[0].legend()

    axes[1].plot(time_axis, mean_trust, 'b-', label='Trust')
    axes[1].fill_between(time_axis, mean_trust - std_trust, mean_trust + std_trust, color='b', alpha=0.3)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Average Trust')
    axes[1].legend()

    axes[2].plot(time_axis, mean_pop, 'k-', label='Population')
    axes[2].fill_between(time_axis, mean_pop - std_pop, mean_pop + std_pop, color='gray', alpha=0.3)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Population Size')
    axes[2].legend()

    plt.tight_layout()
    plt.show()
