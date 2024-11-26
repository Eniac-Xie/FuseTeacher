import logging 
import sys
import oss2
import time
from oss2 import set_stream_logger
set_stream_logger('oss2', logging.FATAL)
logging.basicConfig(level=logging.INFO)

import torch.distributed as dist


class OSSLoggingHandler(logging.StreamHandler):
    def __init__(self, log_file):
        logging.StreamHandler.__init__(self)
        auth = oss2.Auth('your_ak', 'your_sk')
        self._bucket = oss2.Bucket(auth, 'your_endpoint', 'your_bucket_name')
        self._log_file = log_file
        if self._bucket.object_exists(self._log_file):
            self._bucket.delete_object(self._log_file)
        self._pos = self._bucket.append_object(self._log_file, 0, '')

    def emit(self, record):
        try:
            msg = self.format(record) + '\n'
            self._pos = self._bucket.append_object(self._log_file, self._pos.next_position, msg)
        except Exception as err:
            pass


class MultiModalLogging(object):
    def __init__(self):
       self.logger = logging.getLogger()
       self.formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    def add_oss(self, log_dir):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        if rank == 0:
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            ossh = OSSLoggingHandler('{}/{}.log'.format(log_dir, timestamp_str))
            ossh.setLevel(logging.INFO)
            ossh.setFormatter(self.formatter)
            self.logger.addHandler(ossh)

    def add_std(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        if rank == 0:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(self.formatter)
            self.logger.addHandler(ch)

    def get(self):
        return self.logger


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val_list = []
        self.avg = 0

    def update(self, val):
        self.val_list.append(val)
        if len(self.val_list) > 50:
            self.val_list.pop(0)
        self.avg = sum(self.val_list) / len(self.val_list)
