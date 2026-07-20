import carla
import math
import random
import numpy as np
from agents.navigation.behavior_agent import BehaviorAgent
from src.utils import SmoothNoise


class Vehicle:
    """Generic CARLA vehicle with optional autopilot."""

    def __init__(self, world: carla.World, cfg: dict, key: str):
        self.world = world
        self.key = key
        self.cfg = cfg[key]
        self.actor = None
        self.agent = None

        self._spawn()
        self.world.tick()
        self._load_params()

        if self.cfg.get("autopilot", False):
            self.agent = BehaviorAgent(
                self.actor, behavior=self.cfg.get("behavior", "normal")
            )
            if "destination" in self.cfg:
                self._set_destination()
            self.agent.set_target_speed(self.cfg.get("target_speed", 30))
            if self.cfg.get("draw_route", False):
                self.draw_route()

    # ------------------------------------------------------------------
    # Spawn & setup
    # ------------------------------------------------------------------

    def _load_params(self):
        """Load dynamics and noise parameters from config."""
        # dynamics
        self.lr = self.cfg.get("lr", 1.5)
        self.acc_min = self.cfg.get("acc_min", -6.0)
        self.acc_max = self.cfg.get("acc_max", 3.0)
        self.beta_min = self.cfg.get("beta_min", -0.5)
        self.beta_max = self.cfg.get("beta_max", 0.5)
        self.width = self.cfg.get("width", 2.0)
        self.length = self.cfg.get("length", 4.5)
        # noise
        self.std_acc = self.cfg.get("std_acc", 0.1)
        self.std_steer = self.cfg.get("std_steer", 0.02)

    def _spawn(self):
        blueprint = self.world.get_blueprint_library().find(self.cfg["blueprint"])
        if blueprint is None:
            raise ValueError(f"Blueprint '{self.cfg['blueprint']}' not found.")
        blueprint.set_attribute("role_name", self.cfg.get("role", "autopilot"))

        self.actor = self.world.try_spawn_actor(blueprint, self._get_transform("spawn"))
        if self.actor is None:
            raise RuntimeError(f"Failed to spawn at {self.cfg['spawn']}.")

    def _get_transform(self, key: str) -> carla.Transform:
        loc = self.cfg[key]["location"]
        rot = self.cfg[key]["rotation"]
        return carla.Transform(
            carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
            carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"]),
        )

    def _set_destination(self):
        loc = self.cfg["destination"]["location"]
        self.agent.set_destination(carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]))

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def step(self, acc: float = None, steer: float = None):
        """Run one agent step. Optionally override acceleration and/or steering."""
        if self.agent is None:
            raise RuntimeError("step() called but autopilot is disabled.")

        control = self.agent.run_step()
        control.manual_gear_shift = False

        if acc is not None:
            if acc >= 0:
                control.throttle = min(acc, 1.0)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = min(abs(acc), 1.0)

        if steer is not None:
            control.steer = max(-1.0, min(steer, 1.0))

        self.actor.apply_control(control)

    def random_step(self):
        """Run one agent step with smooth noise on acceleration and steering."""
        if self.agent is None:
            raise RuntimeError("random_step() called but autopilot is disabled.")

        if not hasattr(self, "_acc_noise"):
            self._acc_noise = SmoothNoise(mean=0.0, theta=0.3, sigma=self.std_acc)
            self._steer_noise = SmoothNoise(mean=0.0, theta=0.5, sigma=self.std_steer)

        control = self.agent.run_step()
        control.manual_gear_shift = False

        acc = control.throttle + self._acc_noise.sample()
        if acc >= 0:
            control.throttle = max(0.0, min(acc, 1.0))
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acc), 1.0)

        control.steer = max(-1.0, min(control.steer + self._steer_noise.sample(), 1.0))
        self.actor.apply_control(control)

    def apply_control(self, control: carla.VehicleControl):
        """Manually apply control."""
        self.actor.apply_control(control)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        return self.agent is not None and self.agent.done()

    def get_transform(self) -> carla.Transform:
        return self.actor.get_transform()

    def get_velocity(self) -> carla.Vector3D:
        return self.actor.get_velocity()

    def get_speed(self) -> float:
        """Return speed in km/h."""
        v = self.actor.get_velocity()
        return 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_trajectories(self, N: int, dt: float, S: int) -> np.ndarray:
        """Sample S trajectories with smooth noise. Returns (S, N+1, 2)."""
        tf = self.get_transform()
        px0, py0 = tf.location.x, tf.location.y
        yaw0 = math.radians(tf.rotation.yaw)
        speed0 = self.get_speed() / 3.6
        max_steer_rad = math.radians(70.0)

        # no autopilot — return stationary trajectory
        if self.agent is None:
            trajs = np.zeros((S, N + 1, 2))
            trajs[:, :, 0] = px0
            trajs[:, :, 1] = py0
            return trajs

        control = self.agent.run_step()
        base_throttle = control.throttle
        base_steer = control.steer

        trajs = np.zeros((S, N + 1, 2))

        for s in range(S):
            px, py, yaw, speed = px0, py0, yaw0, speed0
            acc_noise = SmoothNoise(mean=0.0, theta=0.3, sigma=self.std_acc)
            steer_noise = SmoothNoise(mean=0.0, theta=0.5, sigma=self.std_steer)

            trajs[s, 0] = [px, py]

            for k in range(N):
                acc = base_throttle + acc_noise.sample()
                if acc >= 0:
                    a = min(acc, 1.0) * self.acc_max
                else:
                    a = -min(abs(acc), 1.0) * abs(self.acc_min)

                steer = max(-1.0, min(base_steer + steer_noise.sample(), 1.0))
                beta = math.atan(0.5 * math.tan(steer * max_steer_rad))

                speed += a * dt
                speed = max(0.0, speed)
                yaw += (speed / self.lr) * beta * dt
                px += speed * math.cos(yaw) * dt
                py += speed * math.sin(yaw) * dt
                trajs[s, k + 1] = [px, py]

        return trajs

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def draw_route(self, life_time: float = 10.0):
        debug = self.world.debug
        for waypoint, _ in self.agent.get_local_planner().get_plan():
            debug.draw_point(
                waypoint.transform.location + carla.Location(z=0.3),
                size=0.08,
                color=carla.Color(0, 255, 0),
                life_time=life_time,
            )

    @property
    def agent_type(self) -> str:
        return "vehicle"


class Walker:
    """Spawns and manages a pedestrian."""

    def __init__(self, world: carla.World, cfg: dict, key: str):
        self.world = world
        self.key = key
        self.cfg = cfg[key]
        self.actor = None

        self._spawn()
        self.world.tick()
        self._load_params()

    # ------------------------------------------------------------------
    # Spawn & setup
    # ------------------------------------------------------------------

    def _load_params(self):
        """Load pedestrian parameters from config."""
        self.mean_speed = self.cfg.get("mean_speed", 1.4)
        self.std_speed = self.cfg.get("std_speed", 0.3)
        self.std_dir = self.cfg.get("std_dir", 0.05)

    def _spawn(self):
        bp = self.world.get_blueprint_library().find(
            self.cfg.get("blueprint", "walker.pedestrian.0001")
        )
        self.actor = self.world.try_spawn_actor(bp, self._get_transform("spawn"))
        if self.actor is None:
            raise RuntimeError(f"Failed to spawn walker at {self.cfg['spawn']}.")

    def _get_transform(self, key: str) -> carla.Transform:
        loc = self.cfg[key]["location"]
        rot = self.cfg[key]["rotation"]
        return carla.Transform(
            carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
            carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
        )

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def step(self, speed: float = None):
        """Walk forward at given speed. If None, use mean_speed."""
        if speed is None:
            speed = self.mean_speed
        control = carla.WalkerControl()
        control.speed = speed
        control.direction = self.actor.get_transform().get_forward_vector()
        self.actor.apply_control(control)

    def random_step(self):
        """Walk with noisy speed and direction."""
        fwd = self.actor.get_transform().get_forward_vector()
        noise_yaw = random.gauss(0, self.std_dir)

        direction = carla.Vector3D(
            x=fwd.x * math.cos(noise_yaw) - fwd.y * math.sin(noise_yaw),
            y=fwd.x * math.sin(noise_yaw) + fwd.y * math.cos(noise_yaw),
            z=0.0
        )

        control = carla.WalkerControl()
        control.speed = max(0.0, random.gauss(self.mean_speed, self.std_speed))
        control.direction = direction
        self.actor.apply_control(control)

    def stop(self):
        self.actor.apply_control(carla.WalkerControl(speed=0.0))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_transform(self) -> carla.Transform:
        return self.actor.get_transform()

    def get_speed(self) -> float:
        """Return speed in km/h."""
        v = self.actor.get_velocity()
        return 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_trajectories(self, N: int, dt: float, S: int) -> np.ndarray:
        """Sample S trajectories with noisy speed and direction. Returns (S, N+1, 2)."""
        tf = self.get_transform()
        px0, py0 = tf.location.x, tf.location.y
        yaw0 = math.radians(tf.rotation.yaw)

        trajs = np.zeros((S, N + 1, 2))

        for s in range(S):
            px, py, yaw = px0, py0, yaw0
            trajs[s, 0] = [px, py]

            for k in range(N):
                yaw += random.gauss(0, self.std_dir)
                speed = max(0.0, random.gauss(self.mean_speed, self.std_speed))
                px += speed * math.cos(yaw) * dt
                py += speed * math.sin(yaw) * dt
                trajs[s, k + 1] = [px, py]

        return trajs

    @property
    def agent_type(self) -> str:
        return "walker"