# https://github.com/google-research/google-research/blob/master/distracting_control/suite.py
# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of MuJoCo-based Reinforcement Learning environments.
The suite provides a similar API to the original dm_control suite.
Users can configure the distractions on top of the original tasks. The suite is
targeted for loading environments directly with similar configurations as those
used in the original paper. Each distraction wrapper can be used independently
though.
"""
try:
  from dm_control import suite  # pylint: disable=g-import-not-at-top
  from dm_control.suite.wrappers import pixels  # pylint: disable=g-import-not-at-top
except ImportError:
  suite = None

from distracting_control import background
from distracting_control import camera
from distracting_control import color
from distracting_control import suite_utils

def is_available():
  return suite is not None


def load(domain_name,
         task_name,
         difficulty=None,
         dynamic=True,
         seed=None,
         background_dataset_path=None,
         background_dataset_videos="train",
         background_kwargs=None,
         camera_kwargs=None,
         color_kwargs=None,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False):
  """Returns an environment from a domain name, task name and optional settings.
  ```python
  env = suite.load('cartpole', 'balance')
  ```
  Adding a difficulty will configure distractions matching the reference paper
  for easy, medium, hard.
  Users can also toggle dynamic properties for distractions.
  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    task_kwargs: Optional `dict` of keyword arguments for the task.
    difficulty: Difficulty for the suite. One of 'easy', 'medium', 'hard'.
    dynamic: Boolean controlling whether distractions are dynamic or static.
    backgound_dataset_path: String to the davis directory that contains the
      video directories.
    background_dataset_videos: String ('train'/'val') or list of strings of the
      DAVIS videos to be used for backgrounds.
    background_kwargs: Dict, overwrites settings for background distractions.
    camera_kwargs: Dict, overwrites settings for camera distractions.
    color_kwargs: Dict, overwrites settings for color distractions.
    task_kwargs: Dict, dm control task kwargs.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.
  Returns:
    The requested environment.
  """
  if not is_available():
    raise ImportError("dm_control module is not available. Make sure you "
                      "follow the installation instructions from the "
                      "dm_control package.")

  if difficulty not in [None, "easy", "medium", "hard"]:
    raise ValueError("Difficulty should be one of: 'easy', 'medium', 'hard'.")

  task_kwargs = task_kwargs or {}
  if seed is not None:
    task_kwargs['random'] = seed

  env = suite.load(
      domain_name,
      task_name,
      task_kwargs=task_kwargs,
      environment_kwargs=environment_kwargs,
      visualize_reward=visualize_reward)

  # Apply background distractions.
  if difficulty or background_kwargs:
    background_dataset_path = (
        background_dataset_path or suite_utils.DEFAULT_BACKGROUND_PATH)
    final_background_kwargs = dict()
    if difficulty:
      # Get kwargs for the given difficulty.
      num_videos = suite_utils.DIFFICULTY_NUM_VIDEOS[difficulty]
      final_background_kwargs.update(
          suite_utils.get_background_kwargs(domain_name, num_videos, dynamic,
                                            background_dataset_path,
                                            background_dataset_videos))
    else:
      # Set the dataset path and the videos.
      final_background_kwargs.update(
          dict(
              dataset_path=background_dataset_path,
              dataset_videos=background_dataset_videos))
    if background_kwargs:
      # Overwrite kwargs with those passed here.
      final_background_kwargs.update(background_kwargs)
    print(final_background_kwargs)
    env = background.DistractingBackgroundEnv(env, **final_background_kwargs)

  # Apply camera distractions.
  if difficulty or camera_kwargs:
    final_camera_kwargs = dict()
    final_camera_kwargs["camera_id"] = 2 if domain_name == "quadruped" else 0
    if difficulty:
      # Get kwargs for the given difficulty.
      scale = suite_utils.DIFFICULTY_SCALE[difficulty]
      final_camera_kwargs.update(
          suite_utils.get_camera_kwargs(domain_name, scale, dynamic))
    if camera_kwargs:
      # Overwrite kwargs with those passed here.
      final_camera_kwargs.update(camera_kwargs)
    env = camera.DistractingCameraEnv(env, **final_camera_kwargs)

  # Apply color distractions.
  if difficulty or color_kwargs:
    final_color_kwargs = dict()
    if difficulty:
      # Get kwargs for the given difficulty.
      scale = suite_utils.DIFFICULTY_SCALE[difficulty]
      final_color_kwargs.update(suite_utils.get_color_kwargs(scale, dynamic))
    if color_kwargs:
      # Overwrite kwargs with those passed here.
      final_color_kwargs.update(color_kwargs)
    env = color.DistractingColorEnv(env, **final_color_kwargs)

  return env