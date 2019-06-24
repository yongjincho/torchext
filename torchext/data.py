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
import re
import sys
import heapq
import random
import logging
import multiprocessing as mp


class Dataset:
    def __iter__(self):
        raise NotImplemented

    def map(self, map_fn, num_workers=2):
        return MappedDataset(self, map_fn, num_workers)

    def filter(self, filter_fn):
        return FilteredDataset(self, filter_fn)

    def repeat(self, count=None):
        return RepeatedDataset(self, count)

    def shuffle(self, buffer_size):
        return ShuffledDataset(self, buffer_size)

    def batch(self, batch_size, collate_fn):
        return BatchedDataset(self, batch_size, collate_fn)

    def bucket(self, boundaries, batch_sizes, length_fn, collate_fn):
        return BucketDataset(self, boundaries, batch_sizes, length_fn, collate_fn)


class Result:
    def __init__(self, job_id, data):
        self.job_id = job_id
        self.data = data

    def __lt__(self, other):
        return self.job_id < other.job_id

    def __eq__(self, other):
        return self.job_id == other.job_id


class Worker(mp.Process):
    def __init__(self, job_queue, result_queue, target):
        super().__init__(daemon=True)
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.target = target

    def run(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break
            job_id, job = job
            data = self.target(job)
            self.result_queue.put(Result(job_id, data))
        self.result_queue.put(None)


class MappedDataset(Dataset):
    def __init__(self, source, map_fn, num_workers):
        self.source = source

        self.job_queue = mp.Queue(maxsize=num_workers * 100)
        self.result_queue = mp.Queue(maxsize=num_workers * 100)

        self.job_assigner = mp.Process(target=self.assign_jobs, daemon=True)

        self.workers = []
        for _ in range(num_workers):
            w = Worker(self.job_queue, self.result_queue, map_fn)
            self.workers.append(w)

        self.results = [] # To sort the result by job id.

        for worker in self.workers:
            worker.start()
        self.job_assigner.start()

    def assign_jobs(self):
        for i, sample in enumerate(self.source):
            self.job_queue.put((i, sample))

        for _ in range(len(self.workers)):
            self.job_queue.put(None)

    def __iter__(self):
        next_job_id = 0
        end_count = 0
        while end_count < len(self.workers):
            if self.results and self.results[0].job_id == next_job_id:
                result = heapq.heappop(self.results)
                yield result.data
                next_job_id += 1
            else:
                result = self.result_queue.get()
                if result is None:
                    end_count += 1
                else:
                    heapq.heappush(self.results, result)
        assert len(self.results) == 0


class FilteredDataset(Dataset):
    def __init__(self, source, filter_fn):
        self.source = source
        self.filter_fn = filter_fn

    def __iter__(self):
        for sample in self.source:
            if self.filter_fn(sample):
                yield sample


class RepeatedDataset(Dataset):
    def __init__(self, source, count=None):
        self.source = source
        self.count = count

    def __iter__(self):
        c = 0
        while self.count is None or c < self.count:
            for sample in self.source:
                yield sample
            c += 1


class ShuffledDataset(Dataset):
    def __init__(self, source, buffer_size):
        self.source = source
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for sample in self.source:
            buf.append(sample)
            if len(buf) == self.buffer_size:
                random.shuffle(buf)
                for sample in buf:
                    yield sample
                buf = []

        if buf:
            for sample in buf:
                yield sample


class BatchedDataset(Dataset):
    def __init__(self, source, batch_size, collate_fn):
        self.source = source
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __iter__(self):
        batch = []
        for sample in self.source:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        if batch:
            yield self.collate_fn(batch)


class BucketDataset(Dataset):
    """tensor2tensor's bucketing method"""

    def __init__(self, source, boundaries, batch_sizes, length_fn, collate_fn):
        self.source = source
        self.boundaries = boundaries
        self.batch_sizes = batch_sizes
        self.length_fn = length_fn
        self.collate_fn = collate_fn

    def __iter__(self):
        buckets = [[] for i in range(len(self.boundaries))]
        for sample in self.source:
            length = self.length_fn(sample)
            for i, boundary in enumerate(self.boundaries):
                if length <= boundary:
                    buckets[i].append(sample)
                    if len(buckets[i]) == self.batch_sizes[i]:
                        yield self.collate_fn(buckets[i])
                        buckets[i] = []
                    break

        for bucket in buckets:
            if bucket:
                yield self.collate_fn(bucket)


class TextLineDataset(Dataset):
    def __init__(self, filename):
        if not filename:
            logging.info("'stdin' will be used for dataset because a file wasn't given.")
        self.filename = filename
        self.newline_pattern = re.compile(r'\r?\n$')

    def __iter__(self):
        if not self.filename:
            f = sys.stdin
        else:
            f = open(self.filename)
        for line in f:
            yield self.newline_pattern.sub('', line)
