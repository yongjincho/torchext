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
import logging
import time
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter

from . import utils, config


class TrainingStats:
    TIMER_KEY = "__timer__"

    def __init__(self):
        self._samples = defaultdict(list)
        self._start = None

    def add(self, name, value):
        self._samples[name].append(value)

    def get(self, name):
        values = self._samples[name]
        if values:
            avg = sum(values) / len(values)
        else:
            avg = 0.0
        self._samples.pop(name)
        return avg

    def __iter__(self):
        names = list(self._samples.keys())
        for name in names:
            yield name, self.get(name)

    def start_timer(self):
        self._start = time.time()

    def stop_timer(self):
        if self._start is None:
            return
        elapsed = time.time() - self._start
        self._samples[self.TIMER_KEY].append(elapsed)

    def get_timer(self):
        sec_per_step = self.get(self.TIMER_KEY)
        return 1 / sec_per_step


class Estimator:
    def __init__(self, model, optimizer, model_dir):
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        logging.info(self.model)

        self.optimizer = optimizer

        self.step = 0
        self.model_dir = model_dir

    def save(self):
        states = {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        utils.save_checkpoint(self.model_dir, self.step, states, config.keep_checkpoint_max)

    def restore(self):
        states = utils.load_checkpoint(self.model_dir)
        if states:
            logging.info("Restoring the saved states...")
            self.step = states["step"] + 1
            self.model.load_state_dict(states["model"])
            self.optimizer.load_state_dict(states["optimizer"])

    def train(self, train_dataset, eval_dataset=None):
        stats = TrainingStats()
        summary = SummaryWriter(log_dir=self.model_dir)
        if eval_dataset:
            eval_summary = SummaryWriter(log_dir=os.path.join(self.model_dir, "eval"))

        for batch in train_dataset:
            stats.start_timer()

            # Forward
            self.model.train()
            _, loss = self.model(**batch)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            stats.stop_timer()

            # Log summary
            stats.add("loss", loss.item())
            if self.step % config.summary_interval == 0:
                avg_loss = stats.get("loss")
                step_per_sec = stats.get_timer()
                summary.add_scalar("loss", avg_loss, self.step)
                summary.add_scalar("train/step_per_sec", step_per_sec, self.step)
                logging.info("step={} loss={:.3f} step/sec={:.3f}".format(self.step, avg_loss, step_per_sec))

            # Evaluate
            if eval_dataset and config.evaluation_interval \
                    and self.step % config.evaluation_interval == 0:
                self.evaluate(eval_dataset, eval_summary)

            # Save a checkpoint
            if config.checkpoint_interval and self.step % config.checkpoint_interval == 0:
                self.save()

            if config.train_steps and self.step == config.train_steps:
                break

            self.step += 1

    def evaluate(self, dataset, summary=None):
        stats = TrainingStats()

        for step, batch in zip(utils.range_step(1, config.eval_steps), dataset):
            with torch.no_grad():
                self.model.eval()
                predictions, loss = self.model(**batch)

            # These values will be averaged at the end of evaluation.
            stats.add("loss", loss.item())
            metrics = self.evaluate_hook(step, predictions, **batch)
            if metrics:
                for name, value in metrics.items():
                    stats.add(name, value)

            logging.info("Evaluation step [{}/{}]".format(step, config.eval_steps))

        # Write loss and metrics to tensorboard and log.
        avg_loss = stats.get("loss")
        if summary:
            summary.add_scalar("loss", avg_loss, self.step)
        metrics_msg = ""
        for name, value in stats:
            summary.add_scalar("metrics/" + name, value, self.step)
            metrics_msg += " {}={:.3f}".format(name, value)

        logging.info("Evaluation result: loss={:.3f}{}".format(avg_loss, metrics_msg))

    def evaluate_hook(self, step, predictions, **kwargs):
        pass

    def predict(self):
        raise NotImplemented
