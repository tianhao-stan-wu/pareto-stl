import carla
from src.utils import set_spectator_transform


class Client:
    """Manages connection to the CARLA server and world configuration."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["carla"]
        self.client = None
        self.world = None
        self.original_settings = None
        self._connect()

    def _connect(self):
        """Connect to CARLA server and load world."""
        self.client = carla.Client(self.cfg["host"], self.cfg["port"])
        self.client.set_timeout(self.cfg["timeout"])

        self.world = self.client.load_world(self.cfg["map"])
        self.original_settings = self.world.get_settings()

        self._apply_settings()

        loc = self.cfg["spectator"]["location"]
        rot = self.cfg["spectator"]["rotation"]
        spectator = carla.Transform(
            carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
            carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
        )
        set_spectator_transform(self.world, spectator)

        # traffic_manager = self.client.get_trafficmanager(self.cfg["port"])
        # # Make all vehicles ignore all traffic lights
        # traffic_manager.ignore_traffic_lights(None, True)


    def _apply_settings(self):
        """Apply sync mode and physics settings."""
        settings = self.world.get_settings()
        settings.synchronous_mode = self.cfg.get("sync_mode", True)
        settings.fixed_delta_seconds = self.cfg.get("dt", 0.05)
        self.world.apply_settings(settings)

    def tick(self):
        """Advance simulation by one tick (sync mode only)."""
        self.world.tick()

    def get_traffic_manager(self, port: int = 8000) -> carla.TrafficManager:
        tm = self.client.get_trafficmanager(port)
        tm.set_synchronous_mode(True)
        return tm

    def quit(self, destroy=False):
        """Restore original world settings on exit."""
        if self.world and self.original_settings:
            self.world.apply_settings(self.original_settings)

        if destroy:

            actor_list = self.world.get_actors()
            # Filter and destroy specific types (e.g., vehicles and sensors)
            for actor in actor_list:
                if "vehicle" in actor.type_id or "walker" in actor.type_id:
                    actor.destroy()


