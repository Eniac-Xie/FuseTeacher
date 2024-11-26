import oss2
import io
import os
import tempfile

import torch
from utils.logging import MultiModalLogging

logging = MultiModalLogging()
logger = logging.get()

def get_bucket(name='your_bucket_name', acc=False):
    if name == 'your_bucket_name':
        endpoint = 'your_endpoint'
        auth = oss2.Auth('your_ak', 'your_sk')
        bucket = oss2.Bucket(auth, endpoint, name)
    else:
        raise ValueError
    return bucket

class OssProxy(object):
    def __init__(self):
        auth = oss2.Auth('your_ak', 'your_sk')
        self.bucket = oss2.Bucket(auth, 'your_endpoint', 'your_bucket_name')

    def download(self, oss_key, verbose=True):
        try:
            if os.path.exists(oss_key):
                return open(oss_key, 'rb').read()
            else:
                if verbose:
                    logger.info('getting oss file: {}'.format(oss_key))
                content = self.bucket.get_object(oss_key).read()
                if verbose:
                    logger.info('get oss file: {} success, size: {}'.format(oss_key, len(content)))
                return content

        except Exception as err:
            logger.error('get oss file: {} error:\n{}'.format(oss_key, err), exc_info=True)

    def upload(self, oss_key, content):
        try:
            result = self.bucket.put_object(oss_key, content)
            logger.info('put oss file: {} success'.format(oss_key))
        except Exception as err:
            logger.error('put oss file: {} error:\n{}'.format(oss_key, err), exc_info=True)

def save_model_to_oss(save_path, ddp_model):
    model_buf = io.BytesIO()
    oss_proxy = OssProxy()
    torch.save(ddp_model.state_dict(), model_buf)
    oss_proxy.upload(save_path, model_buf.getvalue())
    model_buf.close()
