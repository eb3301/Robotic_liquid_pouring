# Import delle librerie necessarie
import torch  # Libreria per il calcolo numerico e il deep learning
import math   # Funzioni matematiche di base
import genesis as gs  # Import del framework Genesis per la simulazione robotica
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat  # Funzioni per la manipolazione di quaternioni

# Funzione di utilità per generare numeri casuali float tra lower e upper
# con una certa shape e su un certo device (CPU/GPU)
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

# Classe principale dell'ambiente di simulazione per il robot Go2
class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        # Numero di ambienti simulati in parallelo
        self.num_envs = num_envs
        # Numero di osservazioni per ambiente
        self.num_obs = obs_cfg["num_obs"]
        # Osservazioni privilegiate (non usate qui)
        self.num_privileged_obs = None
        # Numero di azioni (gradi di libertà controllabili)
        self.num_actions = env_cfg["num_actions"]
        # Numero di comandi (es. velocità desiderate)
        self.num_commands = command_cfg["num_commands"]
        # Device su cui lavorare (CPU o GPU)
        self.device = gs.device

        # Simula la latenza reale del robot (1 step di ritardo)
        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        # Intervallo di tempo tra due step di controllo (50Hz)
        self.dt = 0.02  # control frequency on real robot is 50hz
        # Numero massimo di step per episodio
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # Salva i dizionari di configurazione
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        # Scale per normalizzare osservazioni e reward
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # --- Creazione della scena di simulazione ---
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),  # Opzioni di simulazione
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),  # FPS massimo per la visualizzazione
                camera_pos=(2.0, 0.0, 2.5),  # Posizione della camera
                camera_lookat=(0.0, 0.0, 0.5),  # Punto osservato dalla camera
                camera_fov=40,  # Campo visivo della camera
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),  # Quali ambienti visualizzare
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,  # Risolutore dei vincoli
                enable_collision=True,  # Abilita collisioni
                enable_joint_limit=True,  # Abilita limiti dei giunti
            ),
            show_viewer=show_viewer,  # Mostra o meno la finestra grafica
        )

        # --- Aggiunta del piano fisso come terreno ---
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # --- Inizializzazione della posizione e orientamento iniziali del robot ---
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)  # Quaternione inverso per trasformazioni
        # --- Aggiunta del robot Go2 alla scena ---
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # --- Costruzione della scena per il numero di ambienti specificato ---
        self.scene.build(n_envs=num_envs)

        # --- Ottiene gli indici dei gradi di libertà (DoF) dei motori specificati ---
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # --- Imposta i guadagni proporzionali (kp) e derivativi (kd) per il controllo dei motori ---
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # --- Prepara le funzioni di reward e i buffer per la somma dei reward per episodio ---
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt  # Scala il peso per il dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)  # Collega la funzione di reward
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)  # Buffer reward

        # --- Inizializzazione di tutti i buffer necessari per la simulazione ---
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # Velocità lineare base
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # Velocità angolare base
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # Gravità proiettata
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )  # Vettore gravità globale
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)  # Buffer osservazioni
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)  # Buffer reward
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)  # Buffer reset
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)  # Buffer lunghezza episodio
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)  # Buffer comandi
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )  # Scale per i comandi
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)  # Azioni attuali
        self.last_actions = torch.zeros_like(self.actions)  # Azioni precedenti
        self.dof_pos = torch.zeros_like(self.actions)  # Posizione dei giunti
        self.dof_vel = torch.zeros_like(self.actions)  # Velocità dei giunti
        self.last_dof_vel = torch.zeros_like(self.actions)  # Velocità dei giunti precedente
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # Posizione base
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)  # Quaternione base
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )  # Posizione di default dei giunti
        self.extras = dict()  # Informazioni extra per logging/debug
        self.extras["observations"] = dict()

    # Funzione per estrarre nuovi comandi casuali per gli ambienti specificati
    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

    # Funzione principale che esegue un passo di simulazione
    def step(self, actions):
        # Clippa le azioni tra i valori consentiti
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # Applica la latenza se necessario
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        # Calcola la posizione target dei giunti
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        # Manda il comando al robot
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        # Avanza la simulazione di un passo
        self.scene.step()

        # Aggiorna i buffer di stato
        self.episode_length_buf += 1  # Incrementa il contatore degli step
        self.base_pos[:] = self.robot.get_pos()  # Aggiorna la posizione della base
        self.base_quat[:] = self.robot.get_quat()  # Aggiorna il quaternione della base
        # Calcola gli angoli di roll, pitch, yaw rispetto all'orientamento iniziale
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)  # Quaternione inverso per trasformazioni
        # Trasforma le velocità nel frame locale della base
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        # Proietta la gravità nel frame locale della base
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        # Aggiorna posizione e velocità dei giunti
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # Ogni tot passi, aggiorna i comandi
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # Controllo terminazione episodio: supera lunghezza massima o cade (pitch/roll troppo grandi)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        # Segnala i timeout e resetta gli ambienti che ne hanno bisogno
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Calcola e accumula i reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # Costruisce il vettore di osservazione concatenando vari stati normalizzati
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        # Aggiorna buffer azioni e velocità
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # Salva osservazioni per logging
        self.extras["observations"]["critic"] = self.obs_buf

        # Ritorna osservazioni, reward, reset e info extra
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # Ritorna le osservazioni e info extra
    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    # Placeholder per osservazioni privilegiate (non usato qui)
    def get_privileged_observations(self):
        return None

    # Resetta tutti gli stati e buffer degli ambienti specificati
    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # Resetta posizione e velocità dei giunti
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # Resetta posizione e orientamento della base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Resetta buffer azioni, velocità, lunghezza episodio e reset
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # Aggiorna le info extra per logging
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # Estrae nuovi comandi casuali
        self._resample_commands(envs_idx)

    # Resetta tutti gli ambienti
    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ Funzioni di reward ----------------
    # Premia il tracking della velocità lineare desiderata (xy)
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    # Premia il tracking della velocità angolare desiderata (yaw)
    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    # Penalizza la velocità verticale (il robot dovrebbe restare a terra)
    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    # Penalizza cambi troppo bruschi nelle azioni
    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # Penalizza posizioni dei giunti troppo lontane dalla posizione di default
    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    # Penalizza deviazioni dall'altezza target della base
    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

