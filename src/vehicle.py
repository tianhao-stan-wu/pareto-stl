import carla
from agents.navigation.behavior_agent import BehaviorAgent


class Vehicle:
    """Generic CARLA vehicle with optional autopilot."""

    def __init__(self, world: carla.World, cfg: dict, key: str):
        """
        Parameters
        ----------
        world : carla.World
        cfg   : full config dict
        key   : config section name, e.g. "ego_vehicle", "ambulance"
        """
        self.world = world
        self.cfg = cfg[key]
        self.actor = None
        self.agent = None
        self._spawn()
        self.world.tick()

        if self.cfg.get("autopilot", False):
            self.agent = BehaviorAgent(self.actor, behavior=self.cfg.get("behavior", "normal"))

            if "destination" in self.cfg:
                self._set_destination()

            self.agent.set_target_speed(self.cfg.get("target_speed", 30))

            if self.cfg.get("ignore_traffic_lights", False):
                self.agent.ignore_traffic_lights(True)

            if self.cfg.get("ignore_stop_signs", False):
                self.agent.ignore_stop_signs(True) 

            if self.cfg.get("draw_route", False):
                self.draw_route()

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
            carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
        )

    def _set_destination(self):
        loc = self.cfg["destination"]["location"]
        self.agent.set_destination(carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]))

    def draw_route(self, life_time: float = 10.0):
        debug = self.world.debug

        for waypoint, _ in self.agent.get_local_planner().get_plan():
            debug.draw_point(
                waypoint.transform.location + carla.Location(z=0.3),
                size=0.08,
                color=carla.Color(0, 255, 0),
                life_time=life_time,
            )

    def step(self, acc: float = None):
        if self.agent is None:
            raise RuntimeError("step() called but autopilot is disabled.")
            
        control = self.agent.run_step()
        control.manual_gear_shift = False

        if acc is not None:
            print("Not none")
            if acc >= 0:
                control.throttle = min(acc, 1.0)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = min(abs(acc), 1.0)
                print("break")
        print(control.throttle)
        self.actor.apply_control(control)

    def apply_control(self, control: carla.VehicleControl):
        self.actor.apply_control(control)

    def is_done(self) -> bool:
        return self.agent is not None and self.agent.done()

    def get_transform(self) -> carla.Transform:
        return self.actor.get_transform()

    def get_velocity(self) -> carla.Vector3D:
        return self.actor.get_velocity()

    def get_speed(self) -> float:
        v = self.actor.get_velocity()
        return 3.6 * (v.x**2 + v.y**2 + v.z**2)**0.5