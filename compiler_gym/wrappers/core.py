# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Iterable, Optional, Tuple, Union

from gym import Wrapper
from gym.spaces import Space

from compiler_gym.envs import Env
from compiler_gym.spaces.reward import Reward
from compiler_gym.util.gym_type_hints import (
    ActionType,
    ObservationType,
    RewardType,
    StepType,
)
from compiler_gym.views import ObservationSpaceSpec


class CompilerEnvWrapper(Env, Wrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.Env>` environment
    to allow a modular transformation.

    This class is the base class for all wrappers. This class must be used
    rather than :code:`gym.Wrapper` to support the CompilerGym API extensions
    such as the :code:`fork()` method.
    """

    def __init__(self, env: Env):
        """Constructor.

        :param env: The environment to wrap.

        :raises TypeError: If :code:`env` is not a :class:`CompilerEnv
            <compiler_gym.envs.CompilerEnv>`.
        """
        # No call to gym.Wrapper superclass constructor here because we need to
        # avoid setting the observation_space member variable, which in the
        # CompilerEnv class is a property with a custom setter. Instead we set
        # the observation_space_spec directly.
        self.env = env

    def step(self, action, observations=None, rewards=None):
        return self.env.step(action, observations=observations, rewards=rewards)

    def reset(self, *args, **kwargs) -> ObservationType:
        return self.env.reset(*args, **kwargs)

    def fork(self) -> Env:
        return type(self)(env=self.env.fork())

    @property
    def reward_range(self) -> Tuple[float, float]:
        return self.env.reward_range

    @reward_range.setter
    def reward_range(self, value: Tuple[float, float]):
        self.env.reward_range = value

    @property
    def observation_space(self):
        return self.env.observation_space

    @observation_space.setter
    def observation_space(
        self, observation_space: Optional[Union[str, ObservationSpaceSpec]]
    ) -> None:
        self.env.observation_space = observation_space

    @property
    def observation_space_spec(self):
        return self.env.observation_space_spec

    @observation_space_spec.setter
    def observation_space_spec(
        self, observation_space_spec: Optional[ObservationSpaceSpec]
    ) -> None:
        self.env.observation_space_spec = observation_space_spec

    @property
    def reward_space(self) -> Optional[Reward]:
        return self.env.reward_space

    @reward_space.setter
    def reward_space(self, reward_space: Optional[Union[str, Reward]]) -> None:
        self.env.reward_space = reward_space

    @property
    def action_space(self) -> Space:
        return self.env.action_space

    @action_space.setter
    def action_space(self, action_space: Optional[str]):
        self.env.action_space = action_space

    @property
    def spec(self) -> Any:
        return self.env.spec

    @spec.setter
    def spec(self, value: Any):
        self.env.spec = value


class ActionWrapper(CompilerEnvWrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an action space transformation.
    """

    def step(
        self, action: Union[int, Iterable[int]], observations=None, rewards=None
    ) -> StepType:
        return self.env.step(
            self.action(action), observations=observations, rewards=rewards
        )

    def action(self, action):
        """Translate the action to the new space."""
        raise NotImplementedError

    def reverse_action(self, action):
        """Translate an action from the new space to the wrapped space."""
        raise NotImplementedError


class ObservationWrapper(CompilerEnvWrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an observation space transformation.
    """

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        return self.observation(observation)

    def step(self, *args, **kwargs):
        observation, reward, done, info = self.env.step(*args, **kwargs)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        """Translate an observation to the new space."""
        raise NotImplementedError


class RewardWrapper(CompilerEnvWrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an reward space transformation.
    """

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        observation, reward, done, info = self.env.step(*args, **kwargs)
        # Undo the episode_reward update and reapply it once we have transformed
        # the reward.
        #
        # TODO(cummins): Refactor step() so that we don't have to do this
        # recalculation of episode_reward, as this is prone to errors if, say,
        # the base reward returns NaN or an invalid type.
        if reward is not None and self.episode_reward is not None:
            self.unwrapped.episode_reward -= reward
            reward = self.reward(reward)
            self.unwrapped.episode_reward += reward
        return observation, reward, done, info

    def reward(self, reward):
        """Translate a reward to the new space."""
        raise NotImplementedError


class ConversionWrapperEnv(CompilerEnvWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def convert_action_space(self, space: Space) -> Space:
        return space

    def convert_action(self, action: ActionType) -> ActionType:
        return action

    def convert_observation_space(self, space: Space) -> Space:
        return space

    def convert_observation(self, observation: ObservationType) -> ObservationType:
        return observation

    def convert_reward_space(self, space: Reward) -> Reward:
        return space

    def convert_reward(self, reward: RewardType) -> RewardType:
        return reward

    @property
    def action_space(self) -> Space:
        return self.convert_action_space(self.env.action_space)

    @property
    def reward_space(self) -> Optional[Reward]:
        return self.convert_reward_space(self.env.reward_space)

    @property
    def reward_range(self) -> Tuple[float, float]:
        return (
            self.convert_reward(self.env.reward_range[0]),
            self.convert_reward(self.env.reward_range[1]),
        )

    @property
    def observation_space(self) -> Optional[Space]:
        return self.convert_observation_space(self.env.observation_space)

    def reset(self, *args, **kwargs) -> Optional[ObservationType]:
        return self.convert_observation(self.env.reset(*args, **kwargs))

    def step(
        self,
        action: Union[ActionType, Iterable[ActionType]],
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ) -> StepType:
        observation, reward, done, info = self.env.step(
            self.convert_action(action), observations, rewards
        )
        return (
            self.convert_observation(observation),
            self.convert_reward(reward),
            done,
            info,
        )
