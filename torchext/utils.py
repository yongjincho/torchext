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
import os
import glob
import logging
import argparse
import subprocess

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


def get_argument_parser(description=None):
    parser = argparse.ArgumentParser(description)
    parser.add_argument("-m", "--model_dir", type=str, required=True,
            help="The directory for a trained model is saved.")
    parser.add_argument("-c", "--conf", dest="configs", default=[], nargs="*",
            help="A list of configuration items. "
                 "An item is a file path or a 'key=value' formatted string. "
                 "The type of a value is determined by applying int(), float(), and str() "
                 "to it sequencially.")
    return parser


def parse_args(description=None):
    parser = get_argument_parser(description)
    args = parser.parse_args()
    return args


def check_git_hash(model_dir):
    source_dir = os.getcwd()
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logging.warn('"%s" is not a git repository. Therefore, hash value comparison will be skipped.',
                model_dir)
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logging.warn("git hash values are different. {}(saved) != {}(current)".format(
                saved_hash[:8], cur_hash[:8]))
    else:
        open(path, "w").write(cur_hash)


def redirect_log_to_file(model_dir, level=logging.INFO, filename="train.log"):
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')

    h = logging.StreamHandler()
    h.setFormatter(formatter)
    logger.addHandler(h)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setFormatter(formatter)
    logger.addHandler(h)
