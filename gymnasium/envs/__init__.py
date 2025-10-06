"""Registers the internal gym envs then loads the env plugins for module using the entry point."""

from typing import Any

from gymnasium.envs.registration import make, pprint_registry, register, registry, spec


def register_envs():
    # Classic
    # ----------------------------------------

    if "CartPole-v0" not in registry:
        register(
            id="CartPole-v0",
            entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
            vector_entry_point="gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv",
            max_episode_steps=200,
            reward_threshold=195.0,
        )

    if "CartPole-v1" not in registry:
        register(
            id="CartPole-v1",
            entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
            vector_entry_point="gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv",
            max_episode_steps=500,
            reward_threshold=475.0,
        )

    if "MountainCar-v0" not in registry:
        register(
            id="MountainCar-v0",
            entry_point="gymnasium.envs.classic_control.mountain_car:MountainCarEnv",
            max_episode_steps=200,
            reward_threshold=-110.0,
        )

    if "MountainCarContinuous-v0" not in registry:
        register(
            id="MountainCarContinuous-v0",
            entry_point="gymnasium.envs.classic_control.continuous_mountain_car:Continuous_MountainCarEnv",
            max_episode_steps=999,
            reward_threshold=90.0,
        )

    if "Pendulum-v1" not in registry:
        register(
            id="Pendulum-v1",
            entry_point="gymnasium.envs.classic_control.pendulum:PendulumEnv",
            max_episode_steps=200,
        )

    if "Acrobot-v1" not in registry:
        register(
            id="Acrobot-v1",
            entry_point="gymnasium.envs.classic_control.acrobot:AcrobotEnv",
            reward_threshold=-100.0,
            max_episode_steps=500,
        )

    # Phys2d (jax classic control)
    # ----------------------------------------

    if "phys2d/CartPole-v0" not in registry:
        register(
            id="phys2d/CartPole-v0",
            entry_point="gymnasium.envs.phys2d.cartpole:CartPoleJaxEnv",
            vector_entry_point="gymnasium.envs.phys2d.cartpole:CartPoleJaxVectorEnv",
            max_episode_steps=200,
            reward_threshold=195.0,
            disable_env_checker=True,
        )

    if "phys2d/CartPole-v1" not in registry:
        register(
            id="phys2d/CartPole-v1",
            entry_point="gymnasium.envs.phys2d.cartpole:CartPoleJaxEnv",
            vector_entry_point="gymnasium.envs.phys2d.cartpole:CartPoleJaxVectorEnv",
            max_episode_steps=500,
            reward_threshold=475.0,
            disable_env_checker=True,
        )

    if "phys2d/Pendulum-v0" not in registry:
        register(
            id="phys2d/Pendulum-v0",
            entry_point="gymnasium.envs.phys2d.pendulum:PendulumJaxEnv",
            vector_entry_point="gymnasium.envs.phys2d.pendulum:PendulumJaxVectorEnv",
            max_episode_steps=200,
            disable_env_checker=True,
        )

    # Box2d
    # ----------------------------------------

    if "LunarLander-v3" not in registry:
        register(
            id="LunarLander-v3",
            entry_point="gymnasium.envs.box2d.lunar_lander:LunarLander",
            max_episode_steps=1000,
            reward_threshold=200,
        )

    if "LunarLanderContinuous-v3" not in registry:
        register(
            id="LunarLanderContinuous-v3",
            entry_point="gymnasium.envs.box2d.lunar_lander:LunarLander",
            kwargs={"continuous": True},
            max_episode_steps=1000,
            reward_threshold=200,
        )

    if "BipedalWalker-v3" not in registry:
        register(
            id="BipedalWalker-v3",
            entry_point="gymnasium.envs.box2d.bipedal_walker:BipedalWalker",
            max_episode_steps=1600,
            reward_threshold=300,
        )

    if "BipedalWalkerHardcore-v3" not in registry:
        register(
            id="BipedalWalkerHardcore-v3",
            entry_point="gymnasium.envs.box2d.bipedal_walker:BipedalWalker",
            kwargs={"hardcore": True},
            max_episode_steps=2000,
            reward_threshold=300,
        )

    if "CarRacing-v3" not in registry:
        register(
            id="CarRacing-v3",
            entry_point="gymnasium.envs.box2d.car_racing:CarRacing",
            max_episode_steps=1000,
            reward_threshold=900,
        )

    # Toy Text
    # ----------------------------------------

    if "Blackjack-v1" not in registry:
        register(
            id="Blackjack-v1",
            entry_point="gymnasium.envs.toy_text.blackjack:BlackjackEnv",
            kwargs={"sab": True, "natural": False},
        )

    if "FrozenLake-v1" not in registry:
        register(
            id="FrozenLake-v1",
            entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
            kwargs={"map_name": "4x4"},
            max_episode_steps=100,
            reward_threshold=0.70,  # optimum = 0.74
        )

    if "FrozenLake8x8-v1" not in registry:
        register(
            id="FrozenLake8x8-v1",
            entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
            kwargs={"map_name": "8x8"},
            max_episode_steps=200,
            reward_threshold=0.85,  # optimum = 0.91
        )

    if "CliffWalking-v1" not in registry:
        register(
            id="CliffWalking-v1",
            entry_point="gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv",
        )

    if "CliffWalkingSlippery-v1" not in registry:
        register(
            id="CliffWalkingSlippery-v1",
            entry_point="gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv",
            kwargs={"is_slippery": True},
        )

    if "Taxi-v3" not in registry:
        register(
            id="Taxi-v3",
            entry_point="gymnasium.envs.toy_text.taxi:TaxiEnv",
            reward_threshold=8,  # optimum = 8.46
            max_episode_steps=200,
        )

    # Tabular
    # ----------------------------------------

    if "tabular/Blackjack-v0" not in registry:
        register(
            id="tabular/Blackjack-v0",
            entry_point="gymnasium.envs.tabular.blackjack:BlackJackJaxEnv",
            disable_env_checker=True,
        )

    if "tabular/CliffWalking-v0" not in registry:
        register(
            id="tabular/CliffWalking-v0",
            entry_point="gymnasium.envs.tabular.cliffwalking:CliffWalkingJaxEnv",
            disable_env_checker=True,
        )

    # Mujoco
    # ----------------------------------------

    def _raise_mujoco_py_error(*args: Any, **kwargs: Any):
        raise ImportError(
            "The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics)."
        )

    # manipulation

    if "Reacher-v2" not in registry:
        register(id="Reacher-v2", entry_point=_raise_mujoco_py_error)

    if "Reacher-v4" not in registry:
        register(
            id="Reacher-v4",
            entry_point="gymnasium.envs.mujoco.reacher_v4:ReacherEnv",
            max_episode_steps=50,
            reward_threshold=-3.75,
        )

    if "Reacher-v5" not in registry:
        register(
            id="Reacher-v5",
            entry_point="gymnasium.envs.mujoco.reacher_v5:ReacherEnv",
            max_episode_steps=50,
            reward_threshold=-3.75,
        )

    if "Pusher-v2" not in registry:
        register(id="Pusher-v2", entry_point=_raise_mujoco_py_error)

    if "Pusher-v4" not in registry:
        register(
            id="Pusher-v4",
            entry_point="gymnasium.envs.mujoco.pusher_v4:PusherEnv",
            max_episode_steps=100,
            reward_threshold=0.0,
        )

    if "Pusher-v5" not in registry:
        register(
            id="Pusher-v5",
            entry_point="gymnasium.envs.mujoco.pusher_v5:PusherEnv",
            max_episode_steps=100,
            reward_threshold=0.0,
        )

    # balance

    if "InvertedPendulum-v2" not in registry:
        register(id="InvertedPendulum-v2", entry_point=_raise_mujoco_py_error)

    if "InvertedPendulum-v4" not in registry:
        register(
            id="InvertedPendulum-v4",
            entry_point="gymnasium.envs.mujoco.inverted_pendulum_v4:InvertedPendulumEnv",
            max_episode_steps=1000,
            reward_threshold=950.0,
        )

    if "InvertedPendulum-v5" not in registry:
        register(
            id="InvertedPendulum-v5",
            entry_point="gymnasium.envs.mujoco.inverted_pendulum_v5:InvertedPendulumEnv",
            max_episode_steps=1000,
            reward_threshold=950.0,
        )

    if "InvertedDoublePendulum-v2" not in registry:
        register(id="InvertedDoublePendulum-v2", entry_point=_raise_mujoco_py_error)

    if "InvertedDoublePendulum-v4" not in registry:
        register(
            id="InvertedDoublePendulum-v4",
            entry_point="gymnasium.envs.mujoco.inverted_double_pendulum_v4:InvertedDoublePendulumEnv",
            max_episode_steps=1000,
            reward_threshold=9100.0,
        )

    if "InvertedDoublePendulum-v5" not in registry:
        register(
            id="InvertedDoublePendulum-v5",
            entry_point="gymnasium.envs.mujoco.inverted_double_pendulum_v5:InvertedDoublePendulumEnv",
            max_episode_steps=1000,
            reward_threshold=9100.0,
        )

    # runners

    if "HalfCheetah-v2" not in registry:
        register(id="HalfCheetah-v2", entry_point=_raise_mujoco_py_error)

    if "HalfCheetah-v3" not in registry:
        register(id="HalfCheetah-v3", entry_point=_raise_mujoco_py_error)

    if "HalfCheetah-v4" not in registry:
        register(
            id="HalfCheetah-v4",
            entry_point="gymnasium.envs.mujoco.half_cheetah_v4:HalfCheetahEnv",
            max_episode_steps=1000,
            reward_threshold=4800.0,
        )

    if "HalfCheetah-v5" not in registry:
        register(
            id="HalfCheetah-v5",
            entry_point="gymnasium.envs.mujoco.half_cheetah_v5:HalfCheetahEnv",
            max_episode_steps=1000,
            reward_threshold=4800.0,
        )

    if "Hopper-v2" not in registry:
        register(id="Hopper-v2", entry_point=_raise_mujoco_py_error)

    if "Hopper-v3" not in registry:
        register(id="Hopper-v3", entry_point=_raise_mujoco_py_error)

    if "Hopper-v4" not in registry:
        register(
            id="Hopper-v4",
            entry_point="gymnasium.envs.mujoco.hopper_v4:HopperEnv",
            max_episode_steps=1000,
            reward_threshold=3800.0,
        )

    if "Hopper-v5" not in registry:
        register(
            id="Hopper-v5",
            entry_point="gymnasium.envs.mujoco.hopper_v5:HopperEnv",
            max_episode_steps=1000,
            reward_threshold=3800.0,
        )

    if "Swimmer-v2" not in registry:
        register(id="Swimmer-v2", entry_point=_raise_mujoco_py_error)

    if "Swimmer-v3" not in registry:
        register(id="Swimmer-v3", entry_point=_raise_mujoco_py_error)

    if "Swimmer-v4" not in registry:
        register(
            id="Swimmer-v4",
            entry_point="gymnasium.envs.mujoco.swimmer_v4:SwimmerEnv",
            max_episode_steps=1000,
            reward_threshold=360.0,
        )

    if "Swimmer-v5" not in registry:
        register(
            id="Swimmer-v5",
            entry_point="gymnasium.envs.mujoco.swimmer_v5:SwimmerEnv",
            max_episode_steps=1000,
            reward_threshold=360.0,
        )

    if "Walker2d-v2" not in registry:
        register(id="Walker2d-v2", entry_point=_raise_mujoco_py_error)

    if "Walker2d-v3" not in registry:
        register(id="Walker2d-v3", entry_point=_raise_mujoco_py_error)

    if "Walker2d-v4" not in registry:
        register(
            id="Walker2d-v4",
            entry_point="gymnasium.envs.mujoco.walker2d_v4:Walker2dEnv",
            max_episode_steps=1000,
        )

    if "Walker2d-v5" not in registry:
        register(
            id="Walker2d-v5",
            entry_point="gymnasium.envs.mujoco.walker2d_v5:Walker2dEnv",
            max_episode_steps=1000,
        )

    if "Ant-v2" not in registry:
        register(id="Ant-v2", entry_point=_raise_mujoco_py_error)

    if "Ant-v3" not in registry:
        register(id="Ant-v3", entry_point=_raise_mujoco_py_error)

    if "Ant-v4" not in registry:
        register(
            id="Ant-v4",
            entry_point="gymnasium.envs.mujoco.ant_v4:AntEnv",
            max_episode_steps=1000,
            reward_threshold=6000.0,
        )

    if "Ant-v5" not in registry:
        register(
            id="Ant-v5",
            entry_point="gymnasium.envs.mujoco.ant_v5:AntEnv",
            max_episode_steps=1000,
            reward_threshold=6000.0,
        )

    if "Humanoid-v2" not in registry:
        register(id="Humanoid-v2", entry_point=_raise_mujoco_py_error)

    if "Humanoid-v3" not in registry:
        register(id="Humanoid-v3", entry_point=_raise_mujoco_py_error)

    if "Humanoid-v4" not in registry:
        register(
            id="Humanoid-v4",
            entry_point="gymnasium.envs.mujoco.humanoid_v4:HumanoidEnv",
            max_episode_steps=1000,
        )

    if "Humanoid-v5" not in registry:
        register(
            id="Humanoid-v5",
            entry_point="gymnasium.envs.mujoco.humanoid_v5:HumanoidEnv",
            max_episode_steps=1000,
        )

    if "HumanoidStandup-v2" not in registry:
        register(id="HumanoidStandup-v2", entry_point=_raise_mujoco_py_error)

    if "HumanoidStandup-v4" not in registry:
        register(
            id="HumanoidStandup-v4",
            entry_point="gymnasium.envs.mujoco.humanoidstandup_v4:HumanoidStandupEnv",
            max_episode_steps=1000,
        )

    if "HumanoidStandup-v5" not in registry:
        register(
            id="HumanoidStandup-v5",
            entry_point="gymnasium.envs.mujoco.humanoidstandup_v5:HumanoidStandupEnv",
            max_episode_steps=1000,
        )


    # --- For shimmy compatibility
    def _raise_shimmy_error(*args: Any, **kwargs: Any):
        raise ImportError(
            'To use the gym compatibility environments, run `pip install "shimmy[gym-v21]"` or `pip install "shimmy[gym-v26]"`'
        )


    # When installed, shimmy will re-register these environments with the correct entry_point
    if "GymV21Environment-v0" not in registry:
        register(id="GymV21Environment-v0", entry_point=_raise_shimmy_error)
    if "GymV26Environment-v0" not in registry:
        register(id="GymV26Environment-v0", entry_point=_raise_shimmy_error)

# Only call register_envs() if you want registration
register_envs()
