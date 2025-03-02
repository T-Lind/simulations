#!/usr/bin/env python3
"""
City Block Simulation with Adjusted Extrapolation (Edge Correction),
Multi-Parameter Sweeps, and Professional Statistical Analysis

This simulation models a square city grid (with a configurable number of rows/columns)
populated with two types of agents:
  - Walkers: They move between buildings at a fixed speed and, during each journey,
    record how many scooter riders (active, i.e. moving) they see within a visual radius.
  - Scooter riders: They move faster (default 3x as fast as walkers) between buildings.

Agents are considered active (and visible) only when moving. When in a building (waiting),
they are invisible and not counted.

When a walker completes a journey, it calculates the average observed scooter count from
active scooters only, then extrapolates the total active scooter count using an edge-corrected
scaling factor:

    Estimated Scooters = (avg observed count) × (City Area / Effective Observation Area)

The effective observation area is computed as the area of the intersection between the
walker’s visual circle and the city’s rectangular boundary.

The simulation supports parameter sweeps over:
  - Scooter speed
  - Walker visual radius
  - Number of walkers
  - Number of scooters
  - Grid size

For each parameter, several simulation runs are performed and the mean absolute error
(relative to the ground truth active scooter count) is computed. A progress bar tracks
the overall sweep progress, and a combined figure with well-spaced subplots is produced.
Each subplot is annotated with the best (lowest error) value for that parameter.
A summary of the best values is also printed.

Required packages:
  - pygame
  - matplotlib
  - tqdm
  - numpy
  - shapely
  - json (built-in)
"""

import pygame
import random
import math
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from shapely.geometry import Point, box

# --------------------
# Default Simulation Parameters
# --------------------
# Display and grid parameters
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
GRID_SPACING = 100  # spacing between building centers
MARGIN = 50  # margin from screen edge

# Vision parameters
VISUAL_RADIUS = 75  # default walker visual radius (pixels)

# Movement speeds (pixels per second)
WALKER_SPEED = 50
DEFAULT_SCOOTER_SPEED = 150  # default scooter speed (3x walker speed)

# Waiting time (seconds)
WAIT_TIME_MIN = 2
WAIT_TIME_MAX = 5

# Agent counts and grid size
NUM_WALKERS = 500
NUM_SCOOTERS = 100
GRID_SIZE = 10  # city grid: GRID_SIZE x GRID_SIZE buildings

# Simulation timing
DEFAULT_SIMULATION_TIME = 60  # simulation time in seconds

# Visualization colors (R, G, B)
BACKGROUND_COLOR = (255, 255, 255)  # white
BUILDING_COLOR = (200, 200, 200)  # light gray
WALKER_COLOR = (0, 0, 255)  # blue for walkers
SCOOTER_COLOR = (255, 0, 0)  # red for scooters
VISUAL_CIRCLE_COLOR = (0, 255, 0)  # green outline for a walker's visual radius

# Simulation mode parameters
# When SWEEP_MODE is True, the program runs the multi-parameter sweep (non-animated).
SWEEP_MODE = True
ANIMATE = False  # non-animated mode for sweeps
NUM_RUNS_PER_SETTING = 5  # simulation runs per parameter value


# --------------------
# Load Configuration from JSON (if available)
# --------------------
def load_config(config_file="config.json"):
    config = {}
    if os.path.isfile(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading {config_file}: {e}")
    else:
        print("No configuration file found; using default settings.")
    global DEFAULT_SIMULATION_TIME, ANIMATE, SWEEP_MODE, NUM_RUNS_PER_SETTING
    global DEFAULT_SCOOTER_SPEED, VISUAL_RADIUS, NUM_WALKERS, NUM_SCOOTERS, GRID_SIZE
    if "sim_time" in config:
        DEFAULT_SIMULATION_TIME = config["sim_time"]
    if "animate" in config:
        ANIMATE = config["animate"]
    if "sweep_mode" in config:
        SWEEP_MODE = config["sweep_mode"]
    if "runs" in config:
        NUM_RUNS_PER_SETTING = config["runs"]
    if "scooter_speed" in config:
        DEFAULT_SCOOTER_SPEED = config["scooter_speed"]
    if "visual_radius" in config:
        VISUAL_RADIUS = config["visual_radius"]
    if "num_walkers" in config:
        NUM_WALKERS = config["num_walkers"]
    if "num_scooters" in config:
        NUM_SCOOTERS = config["num_scooters"]
    if "grid_size" in config:
        GRID_SIZE = config["grid_size"]
    return config


# --------------------
# Utility Functions
# --------------------
def create_building_positions(grid_size):
    """Generate positions (as pygame.math.Vector2) for a grid_size x grid_size grid of buildings."""
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = MARGIN + j * GRID_SPACING
            y = MARGIN + i * GRID_SPACING
            positions.append(pygame.math.Vector2(x, y))
    return positions


def compute_city_area(building_positions):
    """Compute the area of the city block from the bounding box of building positions."""
    xs = [pos.x for pos in building_positions]
    ys = [pos.y for pos in building_positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return (max_x - min_x) * (max_y - min_y)


def compute_effective_area(pos, radius, city_polygon):
    """
    Given a walker's position (as pygame.math.Vector2) and its visual radius,
    compute the area of the intersection between the walker's circle and the city polygon.
    """
    circle = Point(pos.x, pos.y).buffer(radius, resolution=32)
    intersection = circle.intersection(city_polygon)
    return intersection.area


# --------------------
# Agent Class
# --------------------
class Agent:
    def __init__(self, kind, pos, scooter_speed=DEFAULT_SCOOTER_SPEED):
        """
        kind: 'walker' or 'scooter'
        pos: starting position (pygame.math.Vector2)
        scooter_speed: speed for scooter riders (ignored for walkers)
        """
        self.kind = kind
        self.pos = pos.copy()
        self.state = 'waiting'  # 'waiting' (in building, invisible) or 'moving' (active)
        self.wait_time = random.uniform(WAIT_TIME_MIN, WAIT_TIME_MAX)
        self.destination = None
        self.speed = WALKER_SPEED if kind == 'walker' else scooter_speed
        # For walkers, record per-journey scooter observations.
        self.journey_observations = None

    def update(self, dt, building_positions):
        """Update the agent's state and position over a timestep dt."""
        if self.state == 'waiting':
            self.wait_time -= dt
            if self.wait_time <= 0:
                # Choose a new destination (different from current position)
                self.destination = random.choice(building_positions)
                while self.destination.distance_to(self.pos) < 1e-3:
                    self.destination = random.choice(building_positions)
                self.state = 'moving'
                if self.kind == 'walker':
                    self.journey_observations = []
        elif self.state == 'moving':
            direction = self.destination - self.pos
            distance_to_dest = direction.length()
            if distance_to_dest < self.speed * dt:
                self.pos = self.destination.copy()
                self.state = 'waiting'
                self.wait_time = random.uniform(WAIT_TIME_MIN, WAIT_TIME_MAX)
            else:
                direction.normalize_ip()
                self.pos += direction * self.speed * dt


# --------------------
# Simulation Class
# --------------------
class Simulation:
    def __init__(self, animate=ANIMATE, scooter_speed=DEFAULT_SCOOTER_SPEED,
                 sim_time=DEFAULT_SIMULATION_TIME, visual_radius=VISUAL_RADIUS,
                 grid_size=GRID_SIZE, num_walkers=NUM_WALKERS, num_scooters=NUM_SCOOTERS):
        self.animate = animate
        self.sim_time = sim_time
        self.visual_radius = visual_radius  # walker visual radius
        self.grid_size = grid_size
        self.num_walkers = num_walkers
        self.num_scooters = num_scooters

        self.building_positions = create_building_positions(grid_size)
        self.city_area = compute_city_area(self.building_positions)
        xs = [pos.x for pos in self.building_positions]
        ys = [pos.y for pos in self.building_positions]
        self.city_polygon = box(min(xs), min(ys), max(xs), max(ys))

        self.agents = []
        # Create walkers.
        for _ in range(num_walkers):
            start_pos = random.choice(self.building_positions)
            self.agents.append(Agent('walker', start_pos))
        # Create scooter riders.
        for _ in range(num_scooters):
            start_pos = random.choice(self.building_positions)
            self.agents.append(Agent('scooter', start_pos, scooter_speed=scooter_speed))

        # Collected journey samples:
        # Each sample is a tuple (avg observed count, extrapolated estimate, ground truth active scooters)
        self.samples = []
        if self.animate:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("City Block Simulation")
            self.clock = pygame.time.Clock()

    def update(self, dt):
        """Update all agents and record per-journey data for walkers."""
        for agent in self.agents:
            agent.update(dt, self.building_positions)
        for agent in self.agents:
            if agent.kind == 'walker':
                if agent.state == 'moving':
                    # Count only active scooters (those moving) within visual radius.
                    count = 0
                    for other in self.agents:
                        if other.kind == 'scooter' and other.state == 'moving':
                            if agent.pos.distance_to(other.pos) <= self.visual_radius:
                                count += 1
                    if agent.journey_observations is not None:
                        agent.journey_observations.append(count)
                elif agent.state == 'waiting':
                    if agent.journey_observations is not None and len(agent.journey_observations) > 0:
                        avg_count = sum(agent.journey_observations) / len(agent.journey_observations)
                        effective_area = compute_effective_area(agent.pos, self.visual_radius, self.city_polygon)
                        scaling_factor = self.city_area / effective_area if effective_area > 0 else 0
                        estimated = avg_count * scaling_factor
                        # Ground truth: active scooters (moving) at this moment.
                        active_scooters = sum(1 for a in self.agents if a.kind == 'scooter' and a.state == 'moving')
                        self.samples.append((avg_count, estimated, active_scooters))
                    agent.journey_observations = None

    def draw(self):
        """Draw the city buildings and only the active agents (those moving)."""
        self.screen.fill(BACKGROUND_COLOR)
        for pos in self.building_positions:
            rect_size = 10
            rect = pygame.Rect(pos.x - rect_size / 2, pos.y - rect_size / 2, rect_size, rect_size)
            pygame.draw.rect(self.screen, BUILDING_COLOR, rect)
        # Draw only active (moving) agents.
        for agent in self.agents:
            if agent.state != 'moving':
                continue
            color = WALKER_COLOR if agent.kind == 'walker' else SCOOTER_COLOR
            pygame.draw.circle(self.screen, color, (int(agent.pos.x), int(agent.pos.y)), 5)
            if agent.kind == 'walker':
                pygame.draw.circle(self.screen, VISUAL_CIRCLE_COLOR,
                                   (int(agent.pos.x), int(agent.pos.y)), self.visual_radius, 1)
        pygame.display.flip()

    def run(self, sim_time=None):
        """Run the simulation for sim_time seconds."""
        if sim_time is None:
            sim_time = self.sim_time
        total_time = 0.0
        while total_time < sim_time:
            dt = self.clock.tick(60) / 1000.0 if self.animate else 0.01
            total_time += dt
            if self.animate:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        total_time = sim_time
            self.update(dt)
            if self.animate:
                self.draw()
        if self.animate:
            pygame.quit()
        if self.animate:
            self.report_results()

    def get_average_error(self):
        """Return the mean absolute error over journeys (|extrapolated - ground truth active|)."""
        if not self.samples:
            return 0.0
        errors = [abs(estimated - active) for (_, estimated, active) in self.samples]
        return sum(errors) / len(errors)

    def report_results(self):
        """Print summary statistics based on journey samples."""
        if not self.samples:
            print("No journey data collected.")
            return
        observed_counts = [s[0] for s in self.samples]
        estimated_counts = [s[1] for s in self.samples]
        ground_truths = [s[2] for s in self.samples]
        avg_observed = sum(observed_counts) / len(observed_counts)
        avg_estimated = sum(estimated_counts) / len(estimated_counts)
        avg_ground_truth = sum(ground_truths) / len(ground_truths)
        error = abs(avg_estimated - avg_ground_truth)
        print("Simulation Results (per journey):")
        print(f"  Total journeys recorded: {len(self.samples)}")
        print(f"  Average observed scooter count: {avg_observed:.2f}")
        print(f"  Average extrapolated scooter count: {avg_estimated:.2f}")
        print(f"  Average active scooter count (ground truth): {avg_ground_truth:.2f}")
        print(f"  Average absolute error: {error:.2f}")


# --------------------
# Simulation Run Helper
# --------------------
def run_simulation_and_get_error(sim_params):
    """
    Run one simulation with the given parameters (a dict) and return the average absolute error.
    """
    sim = Simulation(animate=False,
                     scooter_speed=sim_params.get("scooter_speed", DEFAULT_SCOOTER_SPEED),
                     sim_time=sim_params.get("sim_time", DEFAULT_SIMULATION_TIME),
                     visual_radius=sim_params.get("visual_radius", VISUAL_RADIUS),
                     grid_size=sim_params.get("grid_size", GRID_SIZE),
                     num_walkers=sim_params.get("num_walkers", NUM_WALKERS),
                     num_scooters=sim_params.get("num_scooters", NUM_SCOOTERS))
    sim.run(sim_params.get("sim_time", DEFAULT_SIMULATION_TIME))
    return sim.get_average_error()


# --------------------
# Parameter Sweep Function
# --------------------
def sweep_parameter(param_name, values, default_params, runs):
    """
    For the given parameter name and list of values, run the simulation 'runs' times for each value.
    Return a list of tuples (parameter value, mean error, std error).
    """
    results = []
    for val in values:
        errors = []
        for _ in range(runs):
            sim_params = default_params.copy()
            sim_params[param_name] = val
            error = run_simulation_and_get_error(sim_params)
            errors.append(error)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        results.append((val, mean_error, std_error))
    return results


# --------------------
# Combined Multi-Parameter Sweeps and Plotting
# --------------------
def combined_sweeps_and_analysis(sim_time, runs):
    """
    Run parameter sweeps over multiple parameters (scooter_speed, visual_radius, num_walkers,
    num_scooters, grid_size), print statistical analyses, and produce a combined figure with subplots.
    """
    default_params = {
        "scooter_speed": DEFAULT_SCOOTER_SPEED,
        "visual_radius": VISUAL_RADIUS,
        "num_walkers": NUM_WALKERS,
        "num_scooters": NUM_SCOOTERS,
        "grid_size": GRID_SIZE,
        "sim_time": sim_time
    }

    # Define wider parameter ranges.
    sweep_configs = {
        "scooter_speed": [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
        "visual_radius": [25, 50, 75, 100, 125, 150, 175, 200],
        "num_walkers": [10, 30, 50, 70, 90, 110],
        "num_scooters": [5, 10, 15, 20, 25, 30],
        "grid_size": [3, 5, 7, 9, 11]
    }

    sweep_results = {}
    total_iters = sum(len(vals) for vals in sweep_configs.values()) * runs
    pbar = tqdm(total=total_iters, desc="Combined Parameter Sweep")

    for param, values in sweep_configs.items():
        results = []
        for val in values:
            errors = []
            for _ in range(runs):
                sim_params = default_params.copy()
                sim_params[param] = val
                error = run_simulation_and_get_error(sim_params)
                errors.append(error)
                pbar.update(1)
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            results.append((val, mean_error, std_error))
        sweep_results[param] = results
    pbar.close()

    # Print statistical analysis and find best parameter value for each sweep.
    print("\nStatistical Analysis of Parameter Sweeps:")
    best_params = {}
    for param, results in sweep_results.items():
        vals = np.array([r[0] for r in results])
        errors = np.array([r[1] for r in results])
        corr = np.corrcoef(vals, errors)[0, 1]
        print(f"\nParameter: {param}")
        print("  Value\tMean Error\tStd Error")
        for (v, mean_err, std_err) in results:
            print(f"  {v}\t{mean_err:.2f}\t\t{std_err:.2f}")
        print(f"  Pearson correlation coefficient between {param} and error: {corr:.2f}")
        best_idx = np.argmin(errors)
        best_params[param] = results[best_idx]

    print("\nBest parameter values (lowest mean error):")
    for param, (val, mean_err, std_err) in best_params.items():
        print(f"  {param}: {val} (mean error {mean_err:.2f} ± {std_err:.2f})")

    # Set a professional matplotlib style.
    plt.style.use("ggplot")

    # Create combined plots (one subplot per parameter).
    num_plots = len(sweep_configs)
    cols = 2
    rows = (num_plots + 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    axs = axs.flatten()

    for i, (param, results) in enumerate(sweep_results.items()):
        x = [r[0] for r in results]
        y = [r[1] for r in results]
        yerr = [r[2] for r in results]
        axs[i].errorbar(x, y, yerr=yerr, marker='o', capsize=5, linestyle='-', linewidth=1)
        axs[i].set_xlabel(param, fontsize=12)
        axs[i].set_ylabel("Mean Absolute Error", fontsize=12)
        axs[i].set_title(f"Effect of {param} on Error", fontsize=14)
        axs[i].grid(True)
        # Annotate the best (lowest error) value.
        best_val, best_err, _ = best_params[param]
        axs[i].annotate(f"Min: {best_val}\nError: {best_err:.2f}", xy=(best_val, best_err),
                        xytext=(best_val, best_err + 0.05 * max(y)),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        fontsize=10, ha='center')
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    plt.tight_layout(pad=4.0)
    plt.show()


# --------------------
# Main Execution
# --------------------
def main():
    load_config()  # Load configuration from config.json if available
    if SWEEP_MODE:
        combined_sweeps_and_analysis(DEFAULT_SIMULATION_TIME, NUM_RUNS_PER_SETTING)
    else:
        sim = Simulation(animate=ANIMATE, sim_time=DEFAULT_SIMULATION_TIME)
        sim.run()


if __name__ == "__main__":
    main()
