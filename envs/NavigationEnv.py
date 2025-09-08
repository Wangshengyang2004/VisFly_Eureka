import numpy as np
import logging
from habitat_sim.sensor import SensorType
import os
import sys
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
from VisFly.utils.type import TensorDict


def get_along_vertical_vector(base, obj):
    """
    Decompose obj vector into components along and perpendicular to base vector
    BPTT-safe: no in-place operations that break gradient computation
    """
    # Safe norm computation with minimum clipping to avoid zero gradients
    base_norm = th.clamp(base.norm(dim=1, keepdim=True), min=1e-8)
    _ = th.clamp(obj.norm(dim=1, keepdim=True), min=1e-8)

    # Safe division for normalization
    base_normal = base / base_norm
    along_obj_norm = (obj * base_normal).sum(dim=1, keepdim=True)
    along_vector = base_normal * along_obj_norm
    vertical_vector = obj - along_vector
    vertical_obj_norm = th.clamp(vertical_vector.norm(dim=1), min=1e-8)

    # Ensure we return new tensors to avoid in-place modification issues  
    return along_obj_norm.squeeze() + 0.0, vertical_obj_norm + 0.0, base_norm.squeeze() + 0.0


class NavigationEnv(DroneGymEnvsBase):
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = True,
        requires_grad: bool = False,
        random_kwargs: dict = {},
        dynamics_kwargs: dict = {},
        scene_kwargs: dict = {},
        sensor_kwargs: list = [],
        device: str = "cpu",
        target: Optional[th.Tensor] = None,
        max_episode_steps: int = 256,
        tensor_output: bool = False,
    ):
        # Ensure depth sensor is always available for NavigationEnv
        if not sensor_kwargs:
            sensor_kwargs = [
                {
                    "sensor_type": SensorType.DEPTH,
                    "uuid": "depth",
                    "resolution": [64, 64],
                }
            ]

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
        )

        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([15, 0., 1.5] if target is None else target).reshape(1,-1)

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        orientation = self.envs.dynamics._orientation.clone()
        rela = self.target - self.position
        head_target = orientation.world_to_head(rela.T).T
        head_velocity = orientation.world_to_head((self.velocity - 0).T).T
        state = th.hstack([
            head_target / 10,
            self.orientation,
            head_velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)
        return TensorDict({
            "state": state,
            "depth": th.as_tensor(self.sensor_obs["depth"]/10).clamp(max=1)
        })

    def get_success(self) -> th.Tensor:
        # Use parentheses to ensure correct operator precedence and use logical AND for boolean tensors
        # success_pos = (self.position - self.target).norm(dim=1) <= self.success_radius
        # success_vel = self.velocity.norm(dim=1) <= 0.1
        # return success_pos & success_vel
        return th.full((self.num_agent,), False)

    def get_reward(self) -> th.Tensor:
        pos_factor = -0.1 * 1/9 / 2
        # Compute per-agent speed with an extra dimension for safe broadcasting
        speed = self.velocity.detach().norm(dim=1, keepdim=True)
        unit_velocity = self.velocity.detach() / (speed + 1e-8)
        align = (self.direction * unit_velocity).sum(dim=1)
        min_vel = 0.2
        max_vel = 1.0
        # Remove the extra dim for subsequent scalar operations
        scale = (speed.squeeze(1) - min_vel) / (max_vel - min_vel)
        scale = scale.clamp(min=0.0, max=1.0)
        # If vel <= min_vel, no aware reward, scale = 0; If vel >= max_vel, scale = 1; In between, linear
        r_perception_aware = align * scale * 0.02

        # Collision avoidance
        collision_dist = self.collision_vector.norm(dim=1)
        # Distance to obstacle, trigger when distance < 0.8, exponential decay
        r_collision_dist_penalty = -0.01 * th.exp(-2.0 * (collision_dist - 0.8))


        reward = (
                 (self.position - self.target).norm(dim=1) * pos_factor +
                 (self.velocity - 0).norm(dim=1) * -0.002 +
                 (self.angular_velocity - 0).norm(dim=1) * -0.01 +
                 r_perception_aware +
                 r_collision_dist_penalty
        )

        return reward
    

    def get_failure(self) -> th.Tensor:
        return self.is_collision


if __name__ == "__main__":
    import cv2 as cv
    import matplotlib
    # Configure basic logging for this debug run
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    matplotlib.use('Agg')  # Use non-interactive backend for headless

    # Detect headless environment
    import os
    headless = os.environ.get('DISPLAY') is None or os.environ.get('SSH_CONNECTION') is not None

    random_kwargs = {
        "state_generator":
            {
                "class": "Uniform",
                "kwargs": [
                    {"position": {"mean": [15.0, 0.0, 1], "half": [15.0, 10.0, 0.2]}},
                ]
            }
    }
    target = [15.0, 0.0, 1]
    scene_path = "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_box15_wall"
    sensor_kwargs = [{
        "sensor_type": SensorType.DEPTH,
        "uuid": "depth",
        "resolution": [64, 64],
        "position": [0, 0.2, 0.],
    }]
    scene_kwargs = {
        "path": scene_path,
        "render_settings": {
            "mode": "fix",
            "view": "custom",
            "resolution": [1080, 1920],
            # "position": th.tensor([[6., 6.8, 5.5], [6,4.8,4.5]]),
            "position": th.tensor([[9., 0.1, 30.0], [9., 0, 0]]),
            "line_width": 6.,

            "point": th.tensor([[9., 0, 0], [9, 0, 0]]),
            "trajectory": True,
        }
    }
    num_agent = 4
    env = NavigationEnv(
        visual=True,
        num_scene=1,
        num_agent_per_scene=num_agent,
        random_kwargs=random_kwargs,
        scene_kwargs=scene_kwargs,
        sensor_kwargs=sensor_kwargs,
        dynamics_kwargs={}
    )

    env.reset()

    # Video writer setup for headless mode
    video_writer = None
    obs_writer = None
    if headless:
        fourcc = cv.VideoWriter_fourcc(*'avc1')  # Use avc1 for VSCode/browser compatibility
        video_writer = cv.VideoWriter('debug_video.mp4', fourcc, 10.0, (1920, 1080))
        obs_writer = cv.VideoWriter('debug_obs.mp4', fourcc, 10.0, (128, 128))

    t = 0
    max_steps = 500 if headless else float('inf')  # Limit steps in headless mode

    while t < max_steps:
        a = th.rand((num_agent, 4))
        env.step(a)
        # circile position
        # position = th.tensor([[3., 0, 1]]) + th.tensor([[np.cos(t/10), np.sin(t/10), 0]]) * 2
        # rotation = Quaternion.from_euler(th.tensor(t/10.), th.tensor(t/10.), th.tensor(t/10)).toTensor().unsqueeze(0)
        # env.envs.sceneManager.set_pose(position=position, rotation=rotation)
        # env.envs.update_observation()
        # Create debug points and ring for visualization
        # Sphere for random spawn area (position mean)
        spawn_center = th.tensor([random_kwargs["state_generator"]["kwargs"][0]["position"]["mean"]])
        target_pos = th.tensor([target])

        # Create a ring around target (circle points)
        ring_points = []
        ring_radius = 1.0
        for angle in np.linspace(0, 2 * np.pi, 20):
            x = target_pos[0, 0] + ring_radius * np.cos(angle)
            y = target_pos[0, 1] + ring_radius * np.sin(angle)
            z = target_pos[0, 2]
            ring_points.append([x, y, z])
        ring_curve = th.tensor(ring_points).unsqueeze(0)

        debug_points = th.cat([spawn_center, target_pos], dim=0)

        img = env.render(is_draw_axes=True, points=debug_points, curves=ring_curve)
        # print(env.position[0])
        obs = env.sensor_obs["depth"]

        if headless:
            # Save to video files
            if video_writer:
                video_writer.write(cv.cvtColor(img[0], cv.COLOR_RGB2BGR))
            if obs_writer:
                # Handle COLOR sensor observation (RGBA format)
                obs_data = obs[0][0]  # Get first agent's observation
                
                # Debug print to understand the data shape
                if t == 0:
                    logging.info("Obs data shape: %s, dtype: %s", obs_data.shape, obs_data.dtype)
                    logging.info("Obs data min/max: %.3f/%.3f", float(obs_data.min()), float(obs_data.max()))
                
                try:
                    if len(obs_data.shape) == 3 and obs_data.shape[0] >= 3:  # Multi-channel (RGB/RGBA)
                        # Convert to RGB, then to BGR for OpenCV
                        obs_rgb = obs_data[:3]  # Take first 3 channels (RGB)
                        obs_frame = (obs_rgb * 255).clip(0, 255).astype(np.uint8)
                        obs_frame = np.transpose(obs_frame, (1, 2, 0))  # CHW to HWC
                        obs_frame = cv.cvtColor(obs_frame, cv.COLOR_RGB2BGR)
                    else:  # Single channel or other format
                        if len(obs_data.shape) == 2:  # Already HW format
                            obs_frame = (obs_data * 255).clip(0, 255).astype(np.uint8)
                            obs_frame = cv.cvtColor(obs_frame, cv.COLOR_GRAY2BGR)
                        else:  # CHW format, take first channel
                            obs_frame = (obs_data[0] * 255).clip(0, 255).astype(np.uint8)
                            obs_frame = cv.cvtColor(obs_frame, cv.COLOR_GRAY2BGR)
                    
                    # Ensure correct frame size
                    if obs_frame.shape[:2] != (128, 128):
                        obs_frame = cv.resize(obs_frame, (128, 128))
                    
                    obs_writer.write(obs_frame)
                except Exception as e:
                    if t == 0:
                        logging.warning("Error processing obs frame: %s", e)
                        logging.info("Obs data shape: %s", obs_data.shape)
                    # Write a black frame as fallback
                    black_frame = np.zeros((128, 128, 3), dtype=np.uint8)
                    obs_writer.write(black_frame)
        else:
            # Display in windows
            cv.imshow("img", img[0])
            # cv.imshow("obs", np.transpose(obs[0], (1, 2, 0)))
            cv.imshow("obs", obs[0][0])
            cv.waitKey(100)

        t += 1

    # Cleanup
    if headless and video_writer:
        video_writer.release()
        obs_writer.release()
        logging.info("Videos saved: debug_video.mp4, debug_obs.mp4")
    if not headless:
        cv.destroyAllWindows()