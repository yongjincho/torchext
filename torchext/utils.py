# Copyright 2018 Yongjin Cho
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
import glob
import os
import logging
import time
import argparse

import torch


_checkpoint_file_prefix = "checkpoint"


def get_checkpoint_filepaths(model_dir):
    filepaths = glob.glob("{}/{}-*".format(model_dir, _checkpoint_file_prefix))
    filepaths.sort()
    return filepaths


def save_checkpoint(model_dir, step, states, keep_max=None):
    filepath = os.path.join(model_dir, "{}-{:08d}.pt".format(_checkpoint_file_prefix, step))
    logging.info("Saving the checkpoint to {}".format(filepath))
    torch.save(states, filepath)

    filepaths = get_checkpoint_filepaths(model_dir)
    if keep_max and len(filepaths) > keep_max:
        for filepath in filepaths[:len(filepaths) - keep_max]:
            os.remove(filepath)


def load_checkpoint(model_dir):
    filepaths = get_checkpoint_filepaths(model_dir)
    if not filepaths:
        return None
    latest_file = filepaths[-1]
    logging.info("Loading the checkpoint file: {}".format(latest_file))
    return torch.load(latest_file)


def range_step(start, end=None):
    step = start
    while True:
        if end and step > end:
            raise StopIteration
        yield step
        step += 1


def log_level_value(log_level_string):
    """Convert string valued log level to integer"""
    if log_level_string == "WARN":
        return logging.WARN
    elif log_level_string == "INFO":
        return logging.INFO
    elif log_level_string == "DEBUG":
        return logging.DEBUG
    else:
        raise ValueError("Unknown log level: {}".format(log_level_string))


def parse_args(description=None):
    parser = argparse.ArgumentParser(description)
    parser.add_argument("-m", "--model_dir", type=str, required=True,
            help="The directory for a trained model is saved.")
    parser.add_argument("-c", "--configs", default=[], nargs="*",
            help="A list of configuration items. "
                 "An item is a file path or a 'key=value' formatted string. "
                 "The type of a value is determined by applying int(), float(), and str() "
                 "to it sequencially.")
    parser.add_argument("--log", default="INFO", help="WARN | INFO (default) | DEBUG")
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s\t%(asctime)s\t%(message)s",
                        level=log_level_value(args.log))

    return args
