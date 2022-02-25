# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the OpenAI gym interface for compilers."""
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

import gym

from compiler_gym.spaces import Reward
from compiler_gym.util.gym_type_hints import ActionType, ObservationType, StepType
from compiler_gym.views import ObservationSpaceSpec


class Env(gym.Env, ABC):
    @property
    @abstractmethod
    def observation_space_spec(self) -> ObservationSpaceSpec:
        raise NotImplementedError("abstract method")

    @observation_space_spec.setter
    @abstractmethod
    def observation_space_spec(
        self, observation_space_spec: Optional[ObservationSpaceSpec]
    ):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def fork(self) -> "Env":
        """Fork a new environment with exactly the same state.

        This creates a duplicate environment instance with the current state.
        The new environment is entirely independently of the source environment.
        The user must call :meth:`close() <compiler_gym.envs.Env.close>`
        on the original and new environments.

        If not already in an episode, :meth:`reset()
        <compiler_gym.envs.Env.reset>` is called.

        Example usage:

            >>> env = gym.make("llvm-v0")
            >>> env.reset()
            # ... use env
            >>> new_env = env.fork()
            >>> new_env.state == env.state
            True
            >>> new_env.step(1) == env.step(1)
            True

        :return: A new environment instance.
        """
        raise NotImplementedError("abstract method")

    @abstractmethod
    def reset(  # pylint: disable=arguments-differ
        self, *args, **kwargs
    ) -> Optional[ObservationType]:
        """Reset the environment state.

        This method must be called before :func:`step()`.

        :param benchmark: The name of the benchmark to use. If provided, it
            overrides any value that was set during :func:`__init__`, and
            becomes subsequent calls to :code:`reset()` will use this benchmark.
            If no benchmark is provided, and no benchmark was provided to
            :func:`__init___`, the service will randomly select a benchmark to
            use.

        :param action_space: The name of the action space to use. If provided,
            it overrides any value that set during :func:`__init__`, and
            subsequent calls to :code:`reset()` will use this action space. If
            no action space is provided, the default action space is used.

        :return: The initial observation.

        :raises BenchmarkInitError: If the benchmark is invalid. In this case,
            another benchmark must be used.

        :raises TypeError: If no benchmark has been set, and the environment
            does not have a default benchmark to select from.
        """
        raise NotImplementedError("abstract method")

    @abstractmethod
    def step(
        self,
        action: Union[ActionType, Iterable[ActionType]],
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ) -> StepType:
        """Take a step.

        :param action: An action, or a sequence of actions. When multiple
            actions are provided the observation and reward are returned after
            running all of the actions.

        :param observations: A list of observation spaces to compute
            observations from. If provided, this changes the :code:`observation`
            element of the return tuple to be a list of observations from the
            requested spaces. The default :code:`env.observation_space` is not
            returned.

        :param rewards: A list of reward spaces to compute rewards from. If
            provided, this changes the :code:`reward` element of the return
            tuple to be a list of rewards from the requested spaces. The default
            :code:`env.reward_space` is not returned.

        :return: A tuple of observation, reward, done, and info. Observation and
            reward are None if default observation/reward is not set.

        :raises SessionNotFound: If :meth:`reset()
            <compiler_gym.envs.Env.reset>` has not been called.
        """
        raise NotImplementedError("abstract method")
