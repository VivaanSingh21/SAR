import os
os.environ['MUJOCO_GL'] = 'egl'

import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from argument import parse_args
from common.utils import FrameStack, set_seed_everywhere, override_config
from common.make_env import make
from agent.drq import DrQAgent
from auxiliary.sar import SAR
from algo.sac import SAC


def main():
    # === Parse CLI args (from .sh) ===
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda_id}" if args.cuda else "cpu")
    set_seed_everywhere(args.seed)

    # === Load base config YAMLs ===
    config_dir = Path(args.config_dir)
    common_cfg = OmegaConf.load(config_dir / "common.yaml")
    agent_cfg = OmegaConf.load(config_dir / "agent.yaml")[args.agent]
    aux_cfg = OmegaConf.load(config_dir / "auxiliary.yaml")[args.auxiliary] if args.auxiliary else {}
    algo_cfg = OmegaConf.load(config_dir / "algo.yaml")

    # === Merge all configs ===
    config = OmegaConf.merge(common_cfg, agent_cfg, aux_cfg, algo_cfg)

    # === Override config using CLI args ===
    config = override_config(config, args)

    # === Env setup ===
    domain, task = args.env.split('.')[-2:]
    env = make(
        domain_name=domain,
        task_name=task,
        seed=args.seed,
        image_size=config.buffer_params.image_size,
        action_repeat=config.train_params.action_repeat,
        frame_stack=args.frame_stack,
        background_dataset_path=config.setting.background_dataset_path,
        difficulty=config.setting.difficulty,
        dynamic=args.dynamic,
        background=args.background,
        camera=args.camera,
        color=args.color,
        test_background=args.test_background,
        test_camera=args.test_camera,
        test_color=args.test_color,
        num_videos=config.setting.num_videos
    )

    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_range = [float(env.action_space.low.min()), float(env.action_space.high.max())]

    # === Init base RL algo (SAC) ===
    algo = SAC(
        obs_shape=obs_shape,
        action_shape=action_shape,
        action_range=action_range,
        device=device,
        **config.algo_params
    )

    # === Init Auxiliary Task (SAR) ===
    aux_task = SAR(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        **config.auxiliary[args.auxiliary]
    ) if args.auxiliary else None

    # === Init Agent (DrQ) ===
    agent = DrQAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        action_range=action_range,
        device=device,
        base=algo,
        aux_task=aux_task,
        **config.agent_params
    )

    # === Evaluation loop ===
    for ep in range(config.train_params.num_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.sample_action(obs_tensor)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        print(f"[Episode {ep + 1}] Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
