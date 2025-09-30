import torch
import math
import genesis as gs
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class PouringEnv(gym.Env):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        super().__init__()
        # Basic configuration
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-env_cfg["clip_actions"], high=env_cfg["clip_actions"], shape=(self.num_actions,), dtype=np.float32
        )

        # Simulation parameters
        self.dt = 0.02  # 50Hz control frequency
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # Save configuration dictionaries
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        # Scales for normalization
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # Create simulation scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=100,  # Increased substeps for better stability
                gravity=(0, 0, -9.81),
            ),
            sph_options=gs.options.SPHOptions(
                lower_bound=(-1.5, -1.5, 0.0),
                upper_bound=(1.5, 1.5, 2),
                particle_size=0.009,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 960),
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                world_frame_size=0.5,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=False,
                ambient_light=(0.1, 0.1, 0.1),
            ),
            show_viewer=show_viewer,
        )

        # Add plane
        self.scene.add_entity(gs.morphs.Plane())

        # Add UR5e robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file='/home/edo/thesis/LiquidGenesis/urdf/urdf_files_dataset/urdf_files/matlab/ur_description/urdf/ur5e.urdf',
                fixed=True,
                pos=(0, 0, 0),
                euler=(0, 0, 0),
                scale=1.0,
            ),
            material=gs.materials.Rigid(),
        )

        # Get joint indices
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        self.dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]

        # Add container and liquid
        self._setup_container_and_liquid()

        # Build scene
        self.scene.build(n_envs=num_envs)

        # Set robot gains
        self.robot.set_dofs_kp([2000, 1500, 1000, 1000, 50, 50], self.dofs_idx)
        self.robot.set_dofs_kv([100, 100, 100, 100, 20, 20], self.dofs_idx)

        # Initialize reward functions and buffers
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        # Initialize buffers
        self._init_buffers()

    def _setup_container_and_liquid(self):
        # Container setup
        container_scale = 0.001
        container_mesh_path = '/home/edo/thesis/LiquidGenesis/containers/test-Body.obj'
        
        # Add source container
        self.source_container = self.scene.add_entity(
            gs.morphs.Mesh(
                file=container_mesh_path,
                fixed=True,
                pos=(0.0, -0.04, 0.04),
                euler=(-90, 0, 0),
                scale=container_scale,
                decimate=False,
                convexify=False,
                decompose_nonconvex=True,
                coacd_options=gs.options.CoacdOptions(),
                merge_submeshes_for_collision=True,
            ),
            material=gs.materials.Rigid(needs_coup=True),
        )

        # Link container to robot
        self.scene.link_entities(
            self.robot,
            self.source_container,
            parent_link_name="wrist_3_link",
            child_link_name="test-Body_obj_baselink",
        )

        # Add target container
        self.target_container = self.scene.add_entity(
            gs.morphs.Mesh(
                file=container_mesh_path,
                fixed=True,
                pos=(0.5, 0.5, 0.05),
                euler=(0, 0, 0),
                scale=container_scale,
                decimate=False,
                convexify=False,
                decompose_nonconvex=True,
                coacd_options=gs.options.CoacdOptions(),
                merge_submeshes_for_collision=True,
            ),
            material=gs.materials.Rigid(needs_coup=True),
        )

        # Add liquid
        self.liquid = self.scene.add_entity(
            material=gs.materials.SPH.Liquid(
                rho=1000.0,
                stiffness=50000.0,
                exponent=7.0,
                mu=0.001002,
                gamma=0.0728,
                sampler='pbs'
            ),
            morph=gs.morphs.Box(
                pos=(0.0, -0.04, 0.08),
                size=(0.02, 0.02, 0.02),
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.8, 1.0),
                vis_mode='particle',
            ),
        )

    def _init_buffers(self):
        # Initialize all necessary buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.int32)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        
        # Robot state buffers
        self.dof_pos = torch.zeros((self.num_envs, len(self.dofs_idx)), device=self.device, dtype=torch.float32)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float32)
        self.last_actions = torch.zeros_like(self.actions)
        
        # Container and liquid state buffers
        self.source_container_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.source_container_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.target_container_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.liquid_particles = torch.zeros((self.num_envs, 1000, 3), device=self.device, dtype=torch.float32)  # Assuming max 1000 particles
        
        # Default joint positions
        self.default_dof_pos = torch.tensor([0, -np.pi/2, np.pi/2, 0, 0, 0], device=self.device, dtype=torch.float32)
        
        # Extra info for logging
        self.extras = dict()
        self.extras["observations"] = dict()

    def step(self, action):
        # Support both single and batched actions
        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[None, :]
        action = torch.tensor(action, device=self.device, dtype=torch.float32)
        self.actions = torch.clip(action, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        
        # Calculate target joint positions
        target_dof_pos = self.actions * self.env_cfg["action_scale"] + self.default_dof_pos
        
        # Control robot
        self.robot.control_dofs_position(target_dof_pos, self.dofs_idx)
        
        # Step simulation
        self.scene.step()
        
        # Update episode length
        self.episode_length_buf += 1
        
        # Update state buffers
        self._update_state_buffers()
        
        # Check termination conditions
        self._check_termination()
        
        # Calculate rewards
        self._calculate_rewards()
        
        # Build observation vector
        self._build_observation()
        
        # Update action buffer
        self.last_actions[:] = self.actions[:]
        
        obs = self.obs_buf[0].cpu().numpy() if self.num_envs == 1 else self.obs_buf.cpu().numpy()
        reward = float(self.rew_buf[0].cpu().item()) if self.num_envs == 1 else self.rew_buf.cpu().numpy()
        done = bool(self.reset_buf[0].cpu().item()) if self.num_envs == 1 else self.reset_buf.cpu().numpy()
        info = self.extras.copy()
        # Gymnasium step: obs, reward, terminated, truncated, info
        return obs, reward, done, False, info

    def _update_state_buffers(self):
        # Update robot state
        self.dof_pos[:] = self.robot.get_dofs_position(self.dofs_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.dofs_idx)
        
        # Update container states
        self.source_container_pos[:] = self.source_container.get_pos()
        self.source_container_quat[:] = self.source_container.get_quat()
        self.target_container_pos[:] = self.target_container.get_pos()
        
        # Update liquid state
        particles = self.liquid.get_particles()
        if particles is not None:
            # Convert numpy array to torch tensor and move to correct device
            particles_tensor = torch.from_numpy(particles).to(self.device)
            self.liquid_particles[:, :particles.shape[0], :] = particles_tensor

    def _check_termination(self):
        # Check episode length
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        
        # Check if liquid is spilled (particles below certain height)
        spilled = torch.any(self.liquid_particles[:, :, 2] < 0.0, dim=1)
        self.reset_buf |= spilled
        
        # Check if target container is filled (particles in target container)
        target_filled = torch.any(
            torch.norm(self.liquid_particles[:, :, :2] - self.target_container_pos[:, :2].unsqueeze(1), dim=2) < 0.1,
            dim=1
        )
        self.reset_buf |= target_filled

    def _calculate_rewards(self):
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

    def _build_observation(self):
        # Concatenate all relevant state information
        self.obs_buf = torch.cat([
            self.dof_pos * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.source_container_pos * self.obs_scales["container_pos"],
            self.source_container_quat,
            self.target_container_pos * self.obs_scales["container_pos"],
            self.liquid_particles.mean(dim=1) * self.obs_scales["liquid_pos"],
            self.actions,
        ], dim=-1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs = self.obs_buf[0].cpu().numpy() if self.num_envs == 1 else self.obs_buf.cpu().numpy()
        info = {}
        return obs, info

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # Reset robot state
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.dofs_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # Reset containers and liquid
        self._reset_containers_and_liquid(envs_idx)

        # Reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # Update episode info
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def _reset_containers_and_liquid(self, envs_idx):
        # Reset source container position
        self.source_container.set_pos(
            torch.tensor([0.0, -0.04, 0.04], device=self.device).repeat(len(envs_idx), 1),
            envs_idx=envs_idx
        )
        
        # Convert Euler angles to quaternion (w, x, y, z)
        euler_angles = torch.tensor([-90, 0, 0], device=self.device) * (np.pi / 180.0)  # Convert to radians
        cy = torch.cos(euler_angles[2] * 0.5)
        sy = torch.sin(euler_angles[2] * 0.5)
        cp = torch.cos(euler_angles[1] * 0.5)
        sp = torch.sin(euler_angles[1] * 0.5)
        cr = torch.cos(euler_angles[0] * 0.5)
        sr = torch.sin(euler_angles[0] * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        quat = torch.stack([w, x, y, z]).repeat(len(envs_idx), 1)
        self.source_container.set_quat(quat, envs_idx=envs_idx)

        # Reset target container position
        self.target_container.set_pos(
            torch.tensor([0.5, 0.5, 0.05], device=self.device).repeat(len(envs_idx), 1),
            envs_idx=envs_idx
        )

        # Reset liquid
        self._reset_liquid(envs_idx)

    def _reset_liquid(self, envs_idx):
        # Reset liquid particles to initial position
        initial_pos = torch.tensor([0.0, -0.04, 0.08], device=self.device)
        # Convert envs_idx to integer for single environment reset

        #### DA MODIFICARE CON GRIGLIA ####
        if len(envs_idx) == 1:
            # Create a tensor of shape [n_particles, 3] for all particles
            n_particles = self.liquid.get_particles().shape[0]
            pos = initial_pos.repeat(n_particles, 1)
            self.liquid.set_pos(envs_idx[0].item(), pos=pos)
        else:
            # For multiple environments, reset each one individually
            for idx in envs_idx:
                n_particles = self.liquid.get_particles().shape[0]
                pos = initial_pos.repeat(n_particles, 1)
                self.liquid.set_pos(idx.item(), pos=pos)

    # Reward functions
    def _reward_liquid_in_target(self):
        # Reward for liquid particles in target container
        particles_in_target = torch.norm(
            self.liquid_particles[:, :, :2] - self.target_container_pos[:, :2].unsqueeze(1),
            dim=2
        ) < 0.1
        return torch.mean(particles_in_target.float(), dim=1)

    def _reward_liquid_spilled(self):
        # Penalty for spilled liquid
        spilled = torch.any(self.liquid_particles[:, :, 2] < 0.0, dim=1)
        return -spilled.float()

    def _reward_action_smoothness(self):
        # Penalty for jerky movements
        return -torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_joint_limits(self):
        # Penalty for joint limits violation
        joint_limits = torch.tensor([
            [-np.pi, np.pi],  # shoulder_pan_joint
            [-np.pi, 0],      # shoulder_lift_joint
            [-np.pi, np.pi],  # elbow_joint
            [-np.pi, np.pi],  # wrist_1_joint
            [-np.pi, np.pi],  # wrist_2_joint
            [-np.pi, np.pi],  # wrist_3_joint
        ], device=self.device)
        
        violations = torch.sum(
            torch.maximum(
                torch.maximum(joint_limits[:, 0] - self.dof_pos, self.dof_pos - joint_limits[:, 1]),
                torch.zeros_like(self.dof_pos)
            ),
            dim=1
        )
        return -violations 