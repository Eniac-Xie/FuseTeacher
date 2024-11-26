import random
import bisect
import tempfile
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torch.utils.data.distributed import DistributedSampler

from datasets.tar_dataset import get_yfcc15m_llava_trainset
from datasets.utils import MMData, collate_fn77_cn

from utils.oss_op import get_bucket
from utils.logging import MultiModalLogging

logging = MultiModalLogging()
logger = logging.get()


class CustomMultiDataset(Dataset):

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, epoch, dataname2epoch_size, text_tokenizer,  max_words, img_reso, norm_type, buf_info, new_da=False, tar_size=16,
            return_str=False, return_sample_name=False, sample_tag=False, multi_text=False, multi_image=False, slip_image=False):
        super(CustomMultiDataset, self).__init__()
        self.dataset_list = []
        self.tar_size = tar_size

        if 'yfcc15m_llava' in dataname2epoch_size:
            yfcc15m_llava = get_yfcc15m_llava_trainset(
                epoch=epoch,
                epoch_size=dataname2epoch_size['yfcc15m_llava'],
                sample_id_base=310000000000,
                text_tokenizer=text_tokenizer,
                max_words=max_words,
                img_reso=img_reso,
                norm_type=norm_type,
                new_da=new_da,
                return_str=return_str,
                return_sample_name=return_sample_name,
                sample_tag=sample_tag,
                multi_text=multi_text,
                multi_image=multi_image,
                slip_image=slip_image
            )
            self.dataset_list.append(yfcc15m_llava)

        assert len(self.dataset_list) == len(dataname2epoch_size)
        
        self.data_length = sum([len(tmp) for tmp in self.dataset_list])
        logger.info('multi_dataset with length: {}'.format(self.data_length))

        all_data_id_list = list(range(0, self.data_length))
        random.seed(epoch)
        random.shuffle(all_data_id_list)  # shuffle in-place

        self.num_workers = buf_info['num_workers']
        self.buf_num = buf_info['buf_num']
        self.dist_world_size = dist.get_world_size()
        self.dist_global_rank = dist.get_rank()
        self.mp_buf_num = self.buf_num * self.num_workers * self.dist_world_size
        assert self.mp_buf_num % self.tar_size == 0
        self.init_buf_data_id_list = all_data_id_list[:(self.mp_buf_num//self.tar_size)]
        self.data_id_list = all_data_id_list[(self.mp_buf_num//self.tar_size):]
        self.init_buf = False

        self.cumulative_sizes = self.cumsum(self.dataset_list)

    def __len__(self):
        return len(self.data_id_list)

    def get_by_original_idx(self, original_idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, original_idx)
        if dataset_idx == 0:
            sample_idx = original_idx
        else:
            sample_idx = original_idx - self.cumulative_sizes[dataset_idx - 1]
        return [tmp + [dataset_idx, ] for tmp in self.dataset_list[dataset_idx][sample_idx]]

    def __getitem__(self, index):
        if not self.init_buf:
            self.buf_list = []
            local_data_worker_id = get_worker_info().id
            if local_data_worker_id is None:
                # in the main process
                raise ValueError

            global_data_worker_id = local_data_worker_id + self.dist_global_rank * self.num_workers
            global_data_workers_num = self.dist_world_size * self.num_workers
            worker_init_buf_data_id_list = self.init_buf_data_id_list[global_data_worker_id:len(self.init_buf_data_id_list):global_data_workers_num]
            
            print('ddp_worker: {}, {}, {}'.format(self.dist_global_rank, self.data_id_list[0:5], self.init_buf_data_id_list[0:5]), flush=True)

            for original_id in worker_init_buf_data_id_list:
                print('worker: {} add data id {} to buf {}, buf len: {}'.format(global_data_worker_id, original_id, id(self.buf_list), len(self.buf_list)), flush=True)
                rawdata_list = self.get_by_original_idx(original_id)
                if len(rawdata_list) != self.tar_size:
                    raise ValueError()
                self.buf_list.extend(rawdata_list)
            print('worker: {} init buf done, buf len: {}'.format(global_data_worker_id, len(self.buf_list)), flush=True)
            self.init_buf = True

        for try_idx in range(10):
            try:
                original_id = self.data_id_list[index]
                rawdata_list = self.get_by_original_idx(original_id)
                if len(rawdata_list) != self.tar_size:
                    raise ValueError()

                # 加到buffer里
                self.buf_list.extend(rawdata_list)
                random.shuffle(self.buf_list)
                current_list = self.buf_list[:self.tar_size]
                self.buf_list = self.buf_list[self.tar_size:]

                text_list, img_list, sample_id_list = [], [], []
                mmdata_list = []
                for rawdata in current_list:
                    mm_data = self.dataset_list[rawdata[-1]].prepare_mmdata(*rawdata[:3])
                    mmdata_list.append(mm_data)

            except Exception as err:
                print(err, 'original_id: {}'.format(original_id))
                mmdata_list = None
            
            if mmdata_list is not None:
                break
            else:
                index = random.choice(range(len(self.data_id_list)))
        
        assert mmdata_list is not None
        return mmdata_list


def get_multi_dataloaders(batch_size, epoch, dataname2epoch_size, text_tokenizer, max_words, img_reso, norm_type, buf_num, num_workers,
        collate_fn=collate_fn77_cn, new_da=False, tar_size=16, return_str=False, return_sample_name=False, sample_tag=False, multi_text=False,
        multi_image=False, slip_image=False):
    for _ in range(1):
        buf_info = {'num_workers': num_workers, 'buf_num': buf_num}
        train_dataset = CustomMultiDataset(
            epoch=epoch,
            dataname2epoch_size=dataname2epoch_size,
            text_tokenizer=text_tokenizer, 
            max_words=max_words,
            img_reso=img_reso,
            norm_type=norm_type,
            buf_info=buf_info,
            new_da=new_da,
            tar_size=tar_size,
            return_str=return_str,
            return_sample_name=return_sample_name,
            sample_tag=sample_tag,
            multi_text=multi_text,
            multi_image=multi_image,
            slip_image=slip_image)
        
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
        train_sampler.set_epoch(epoch)

        train_params = {
                        'pin_memory': True,
                        'collate_fn': collate_fn,
                        'batch_size': batch_size,
                        'shuffle': False,
                        'drop_last': True,
                        'sampler': train_sampler,
                        'num_workers': num_workers,
                        'prefetch_factor': 1}

        train_loader = DataLoader(train_dataset, **train_params)
        yield (train_loader, 'multi_data')
