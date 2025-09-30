import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env_pouring_rl import PouringEnv
import genesis as gs

# Configurazione dell'ambiente
env_cfg = {
    "num_actions": 6,  # 6 DOF del robot
    "episode_length_s": 10.0,  # durata massima episodio
    "clip_actions": 1.0,  # limite azioni
    "action_scale": 0.1,  # scala azioni
}

obs_cfg = {
    "num_obs": 31,  # dimensione vettore osservazioni
    "obs_scales": {
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "container_pos": 1.0,
        "liquid_pos": 1.0,
    }
}

reward_cfg = {
    "reward_scales": {
        "liquid_in_target": 1.0,
        "liquid_spilled": -1.0,
        "action_smoothness": -0.1,
        "joint_limits": -0.1,
    }
}

command_cfg = {
    "num_commands": 0,  # non usiamo comandi in questo task
}

def make_env():
    """Funzione per creare un'istanza dell'ambiente"""
    env = PouringEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False  # Disabilitiamo il viewer durante il training
    )
    env = Monitor(env)  # Aggiungiamo il monitor per il logging
    return env

def main():
    gs.init(
    backend = gs.cpu,
    logging_level="warning",
)

    # Creiamo la directory per i log e i checkpoint
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Creiamo l'ambiente
    env = DummyVecEnv([make_env])
    
    # Normalizziamo le osservazioni e i reward
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Creiamo l'ambiente di valutazione
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        training=False  # Non aggiorniamo le statistiche durante la valutazione
    )
    
    # Callback per la valutazione
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_results"),
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Callback per il salvataggio dei checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="pouring_model"
    )
    
    # Creiamo il modello PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Aumentiamo l'entropia per favorire l'esplorazione
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=log_dir,
        # policy_kwargs=dict(
        #     net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Architettura della rete neurale
        # ),
        verbose=1
    )
    
    # Training
    total_timesteps = 1000000  # Numero totale di step di training
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Salviamo il modello finale
    model.save(os.path.join(log_dir, "final_model"))
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))

if __name__ == "__main__":
    main() 