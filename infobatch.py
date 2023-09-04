import math
import numpy as np
import copy
from torch.utils.data import Dataset

def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False

class InfoBatch(Dataset):
    def __init__(self, dataset, ratio = 0.5, num_epoch=None, delta = 0.875):
        self.dataset = dataset
        self.ratio = ratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.scores = np.ones([len(self.dataset)])
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0

    def __setscore__(self, indices, values):
        self.scores[indices] = values

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        idx = int(index)
        data = {k:v for k,v in self.dataset[idx].items()}
        data.update({'sample_idx': index, 'weight':self.weights[idx]})
        return data

    def __getitems__(self,index):
        """possibly batched index"""
        # print('InfoBatch.__getitems__ called')
        if has_length(index):
            data = [{k:v for k,v in self.dataset[int(idx)].items()} for idx in index]
            for i,idx in enumerate(index):
                data[i]['sample_idx'] = idx
                data[i]['weight'] = self.weights[int(idx)]
        else: raise ValueError("infobatch __getitems__ got index with no length!")
        # print(data)
        return data

    def __get_wt_id__(self):
        return self.weights[self.last_indexes], self.last_indexes

    def prune(self):
        # prune samples that are well learned, rebalence the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance

        b = self.scores<self.scores.mean()
        well_learned_samples = np.where(b)[0]
        pruned_samples = []
        pruned_samples.extend(np.where(np.invert(b))[0])
        selected = np.random.choice(well_learned_samples, int(self.ratio*len(well_learned_samples)),replace=False)
        self.reset_weights()
        if len(selected)>0:
            self.weights[selected]=1/self.ratio
            pruned_samples.extend(selected)
        print('Cut {} samples for next iteration'.format(len(self.dataset)-len(pruned_samples)))
        self.save_num += len(self.dataset)-len(pruned_samples)
        np.random.shuffle(pruned_samples)
        return pruned_samples

    def pruning_sampler(self):
        return InfoBatchSampler(self, self.num_epoch, self.delta)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.scores.mean()

    def normal_sampler_no_prune(self):
        return InfoBatchSampler(self.no_prune)

    def get_weights(self,indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))



class InfoBatchSampler():
    def __init__(self, infobatch_dataset, num_epoch = math.inf, delta = 1):
        self.infobatch_dataset = infobatch_dataset
        self.seq = None
        self.stop_prune = num_epoch * delta
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed+=1
        if self.seed>self.stop_prune:
            if self.seed <= self.stop_prune+1:
                self.infobatch_dataset.reset_weights()
            self.seq = self.infobatch_dataset.no_prune()
        else:
            self.seq = self.infobatch_dataset.prune()
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self