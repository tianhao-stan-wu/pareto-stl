import carla
import math
import random
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
        self._load_dynamics()

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

    def _load_dynamics(self):
        """Load dynamics parameters from config as attributes."""
        self.lr = self.cfg.get("lr", 1.5)
        self.acc_min = self.cfg.get("acc_min", -6.0)
        self.acc_max = self.cfg.get("acc_max", 3.0)
        self.beta_min = self.cfg.get("beta_min", -0.5)
        self.beta_max = self.cfg.get("beta_max", 0.5)
        self.width = self.cfg.get("width", 2.0)
        self.length = self.cfg.get("length", 4.5)

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

    def step(self, acc: float = None):
        """Run one agent step. Optionally override acceleration."""
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

        self.actor.apply_control(control)

    def random_step(self):
        """Run one agent step with smooth noise on acceleration and steering."""
        if self.agent is None:
            raise RuntimeError("random_step() called but autopilot is disabled.")

        if not hasattr(self, "_acc_noise"):
            self._acc_noise = SmoothNoise(mean=0.0, theta=0.3, sigma=self.cfg.get("std_acc", 0.1))
            self._steer_noise = SmoothNoise(mean=0.0, theta=0.5, sigma=self.cfg.get("std_steer", 0.02))

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

    # ------------------------------------------------------------------
    # Spawn & setup
    # ------------------------------------------------------------------

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
        """Walk forward at given speed. If None, use config mean_speed."""
        if speed is None:
            speed = self.cfg.get("mean_speed", 1.4)
        control = carla.WalkerControl()
        control.speed = speed
        control.direction = self.actor.get_transform().get_forward_vector()
        self.actor.apply_control(control)

    def random_step(self):
        """Walk with noisy speed and direction."""
        fwd = self.actor.get_transform().get_forward_vector()
        noise_yaw = random.gauss(0, self.cfg.get("std_dir", 0.05))

        direction = carla.Vector3D(
            x=fwd.x * math.cos(noise_yaw) - fwd.y * math.sin(noise_yaw),
            y=fwd.x * math.sin(noise_yaw) + fwd.y * math.cos(noise_yaw),
            z=0.0
        )

        control = carla.WalkerControl()
        control.speed = max(0.0, random.gauss(
            self.cfg.get("mean_speed", 1.4),
            self.cfg.get("std_speed", 0.3)
        ))
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
        v = self.actor.get_velocity()
        return 3.6 * (v.x**2 + v.y**2 + v.z**2)**0.5

    @property
    def agent_type(self) -> str:
        return "walker"