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
import csv
import random
import logging


class Dataset:
    def __iter__(self):
        raise NotImplemented

    def map(self, map_fn):
        return MappedDataset(self, map_fn)

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


class MappedDataset(Dataset):
    def __init__(self, source, map_fn):
        self.source = source
        self.map_fn = map_fn

    def __iter__(self):
        for sample in self.source:
            yield self.map_fn(sample)


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
    """Tensor2Tensor's bucketing method"""

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


class CsvDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        reader = csv.DictReader(open(self.filename))
        for row in reader:
            yield row
