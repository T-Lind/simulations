import pygame
import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from tqdm import tqdm

# Configuration
ANIMATE = False  # Set to False to run simulation without animation.
NUM_STEPS = 10000  # Total simulation steps.

# Constants
GRID_SIZE = 100  # Grid dimensions
VISION_RANGE = 5  # Distance agents can see resources
COMM_RANGE = 20  # Communication range (Euclidean)
CLUSTER_NUM = 7  # Number of resource clusters
CLUSTER_STRENGTH = 3  # Peak resource contribution per cluster
CLUSTER_SIGMA = 3  # Spread of resource clusters
RESOURCE_MAX = 4  # Maximum resource per cell
RESOURCE_REGEN = 0.01  # Regeneration rate factor (very low value)
COLLECTION_RATE = 1  # Max resources collected per step
ENERGY_COST = 0.2  # Energy cost per step
REPRODUCTION_THRESHOLD = 50  # Energy needed to reproduce
MUTATION_STD = 0.02  # Standard deviation for trait mutation
RESOURCE_THRESHOLD = 5  # Minimum resource level to trigger signaling
SCREEN_SIZE = 1000  # Pygame window size (pixels)

# Dynamic simulation constants
EXPLORATION_RATE = 0.01  # Base probability an agent randomly explores (scaled by energy)
EVENT_INTERVAL = 100  # Steps between dynamic environmental events
EVENT_RADIUS = 20  # Radius of drought event (in grid cells)
EVENT_INTENSITY = 0.1  # Drought intensity factor (resources multiplied by this factor)

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
        # Initialize cluster centers with continuous positions.
        self.clusters = [{'x': random.uniform(0, GRID_SIZE), 'y': random.uniform(0, GRID_SIZE)}
                         for _ in range(CLUSTER_NUM)]
        self.resources = np.zeros((GRID_SIZE, GRID_SIZE))
        self.update_target_resources()

    def update_target_resources(self) -> None:
        """Compute target resource levels based on cluster positions using vectorized operations."""
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
        """Shift cluster centers slightly, wrapping around grid. Occasionally reposition abruptly."""
        for cluster in self.clusters:
            if random.random() < 0.005:  # 0.5% chance for an abrupt change.
                cluster['x'] = random.uniform(0, GRID_SIZE)
                cluster['y'] = random.uniform(0, GRID_SIZE)
            else:
                cluster['x'] = (cluster['x'] + random.gauss(0, 0.1)) % GRID_SIZE
                cluster['y'] = (cluster['y'] + random.gauss(0, 0.1)) % GRID_SIZE
        self.update_target_resources()

    def update_resources(self) -> None:
        """Adjust resources toward target levels using the regeneration factor and clip to max level."""
        self.resources += RESOURCE_REGEN * (self.target_resources - self.resources)
        self.resources = np.clip(self.resources, 0, RESOURCE_MAX)

    def collect(self, x: int, y: int) -> float:
        """Collect resources from a cell, reducing its level."""
        collected = min(self.resources[x, y], COLLECTION_RATE)
        self.resources[x, y] -= collected
        return collected


@dataclass
class Agent:
    """Represents an agent with position, energy, and evolvable traits."""
    x: int
    y: int
    energy: float
    honesty: float  # Probability of sending true signal
    trust: float  # Probability of trusting received signal
    target: tuple = None  # Target location (tx, ty)

    def see_resources(self, grid: ResourceGrid) -> list:
        """Return list of (position, resource) tuples within vision range."""
        seen = []
        for dx in range(-VISION_RANGE, VISION_RANGE + 1):
            for dy in range(-VISION_RANGE, VISION_RANGE + 1):
                nx = (self.x + dx) % GRID_SIZE
                ny = (self.y + dy) % GRID_SIZE
                seen.append(((nx, ny), grid.resources[nx, ny]))
        return seen

    def find_best_seen(self, grid: ResourceGrid) -> tuple:
        """Find the best resource location within vision; stay if current is best."""
        seen = self.see_resources(grid)
        best_loc, best_val = max(seen, key=lambda item: item[1])
        return best_loc if best_val > grid.resources[self.x, self.y] else (self.x, self.y)

    def send_signal(self, grid: ResourceGrid) -> tuple:
        """
        Send a signal if a high-resource cell is seen.
        Returns either the true best location or a random false location based on honesty.
        """
        visible = [item for item in self.see_resources(grid) if item[1] > RESOURCE_THRESHOLD]
        if visible:
            best_loc, _ = max(visible, key=lambda item: item[1])
            if random.random() < self.honesty:
                return best_loc  # True location.
            return (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        return None

    def move_towards(self, target: tuple) -> tuple:
        """Move one step towards the target on a toroidal grid."""
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
        """Collect resources at the current position and update energy."""
        collected = grid.collect(self.x, self.y)
        self.energy += collected

    def reproduce(self) -> 'Agent':
        """Create an offspring with mutated traits if energy is sufficient."""
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
    """Manages the simulation, agents, and visualization."""

    def __init__(self, num_agents: int = 100) -> None:
        self.grid = ResourceGrid()
        self.agents = [Agent(random.randint(0, GRID_SIZE // 4),
                             random.randint(0, GRID_SIZE // 4),
                             50, random.random(), random.random())
                       for _ in range(num_agents)]
        self.time = 0
        self.data = {'time': [], 'avg_honesty': [], 'avg_trust': [], 'pop_size': []}

    def trigger_drought_event(self) -> None:
        """Simulate a drought event reducing resources in a random region."""
        center_x = random.randint(0, GRID_SIZE - 1)
        center_y = random.randint(0, GRID_SIZE - 1)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                dx = min(abs(x - center_x), GRID_SIZE - abs(x - center_x))
                dy = min(abs(y - center_y), GRID_SIZE - abs(y - center_y))
                if np.sqrt(dx ** 2 + dy ** 2) <= EVENT_RADIUS:
                    self.grid.resources[x, y] *= EVENT_INTENSITY

    def run_step(self) -> None:
        """Execute one simulation step."""
        # Update environment.
        self.grid.move_clusters()
        self.grid.update_resources()

        # Trigger a drought event periodically.
        if self.time != 0 and self.time % EVENT_INTERVAL == 0:
            self.trigger_drought_event()

        # Collect signals from agents.
        signals = []
        for agent in self.agents:
            sig = agent.send_signal(self.grid)
            if sig:
                signals.append((agent.x, agent.y, sig))

        # Agent actions.
        for agent in self.agents:
            # Agents collect resources.
            agent.collect(self.grid)

            # Process received signals using toroidal distance.
            for sx, sy, loc in signals:
                dx = toroidal_distance(sx, agent.x)
                dy = toroidal_distance(sy, agent.y)
                if np.hypot(dx, dy) < COMM_RANGE and random.random() < agent.trust:
                    agent.target = loc

            # Exploration: adjust exploration probability based on energy.
            # Agents with lower energy are less likely to explore.
            exploration_prob = EXPLORATION_RATE * min(1, agent.energy / REPRODUCTION_THRESHOLD)
            if agent.target is None and random.random() < exploration_prob:
                agent.target = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

            # Move agent.
            if agent.target:
                new_pos = agent.move_towards(agent.target)
                if new_pos == agent.target:
                    agent.target = None
                agent.x, agent.y = new_pos
            else:
                best = agent.find_best_seen(self.grid)
                if best != (agent.x, agent.y):
                    agent.x, agent.y = agent.move_towards(best)

            # Update energy due to movement.
            agent.energy -= ENERGY_COST

        # Handle reproduction and death.
        survivors = []
        new_agents = []
        for agent in self.agents:
            if agent.energy > 0:
                survivors.append(agent)
                offspring = agent.reproduce()
                if offspring:
                    new_agents.append(offspring)
        self.agents = survivors + new_agents

        # Record simulation data.
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
        """Render the current simulation state with Pygame."""
        screen.fill((0, 0, 0))
        cell_size = SCREEN_SIZE // GRID_SIZE
        # Draw resources.
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                intensity = int(255 * self.grid.resources[x, y] / RESOURCE_MAX)
                pygame.draw.rect(screen, (0, intensity, 0),
                                 (x * cell_size, y * cell_size, cell_size, cell_size))
        # Draw agents.
        for agent in self.agents:
            color = (int(255 * agent.honesty), 0, int(255 * agent.trust))
            pygame.draw.circle(screen, color,
                               (agent.x * cell_size + cell_size // 2,
                                agent.y * cell_size + cell_size // 2), 2)
        pygame.display.flip()
        clock.tick(30)

    def run(self, steps: int, animate: bool = True) -> None:
        """Run the simulation for a specified number of steps."""
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
        self.plot()

    def plot(self) -> None:
        """Plot evolutionary trends using Matplotlib."""
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
    sim = Simulation(num_agents=100)
    sim.run(steps=NUM_STEPS, animate=ANIMATE)
