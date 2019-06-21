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

# Default system configurations
import logging as _logging


_config_file_name = "config.yml"


def _save(model_dir):
    import os, yaml

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_yml = yaml.dump(_dict(), default_flow_style=False)
    with open(os.path.join(model_dir, _config_file_name), "w+") as f:
        f.write(config_yml)


def _update(key, value):
    g = globals()
    if key in g and g[key] != value:
        _logging.info("The original value of configuration '{}' is '{}', but it is replaced by '{}'.".format(key, g[key], value))
    g[key] = value


def _dict():
    return {k: v for k, v in globals().items() if not k.startswith("_")}


def _print():
    _logging.info("------------------- All configurations --------------------")
    kv = _dict()
    for k in sorted(kv.keys()):
        _logging.info("%s = %s", k, kv[k])
    _logging.info("------------------------------------------------------------")


def _load(model_dir, configs, initialize=False, print_=True):
    import os
    import yaml

    saved_config_file = os.path.join(model_dir, _config_file_name)
    if os.path.exists(saved_config_file):
        configs = [saved_config_file] + configs
    elif not initialize:
        raise ValueError("{} is an invalid model directory.".format(model_dir))

    for cfg in configs:
        kv = [s.strip() for s in cfg.split("=", 1)]
        if len(kv) == 1:
            if not os.path.exists(cfg):
                raise ValueError("The configuration file doesn't exist; {}".format(cfg))
            obj = yaml.load(open(cfg).read(), Loader=yaml.FullLoader)
            for k, v in obj.items():
                _update(k, v)
        else:
            k, v = kv
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    v_norm = v.lower().strip()
                    if v_norm == "true":
                        v = True
                    elif v_norm == "false":
                        v = False
                    elif v_norm == "null":
                        v = None
            _update(k, v)

    if not os.path.exists(saved_config_file) and initialize:
        _save(model_dir)

    if print_:
        _print()
