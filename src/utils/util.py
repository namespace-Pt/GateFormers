import os
import pickle
import pandas as pd
from random import sample
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer



def load_pickle(path):
    """ load pickle file
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def download_plm(bert, dir):
    # initialize bert related parameters
    bert_loading_map = {
        "bert": "bert-base-uncased",
        "deberta": "microsoft/deberta-base",
    }
    os.makedirs(dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(bert_loading_map[bert])
    model = AutoModel.from_pretrained(bert_loading_map[bert])
    tokenizer.save_pretrained(dir)
    model.save_pretrained(dir)


def _group_lists(impr_indexes, *associated_lists):
        """
            group lists by impr_index
        Args:
            associated_lists: list of lists, where list[i] is associated with the impr_indexes[i]

        Returns:
            Iterable: grouped labels (if inputted) and preds
        """
        list_num = len(associated_lists)
        dicts = [defaultdict(list) for i in range(list_num)]

        for x in zip(impr_indexes, *associated_lists):
            key = x[0]
            values = x[1:]
            for i in range(list_num):
                dicts[i][key].extend(values[i])

        grouped_lists = [list(d.values()) for d in dicts]

        return grouped_lists


def sample_news(news, k):
    """ Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number

    Returns:
        list: output of sample list.
        int: count of valid news
    """
    num = len(news)
    if k > num:
        return news + [0] * (k - num), num
    else:
        return sample(news, k), k


def construct_nid2index(news_path, cache_dir):
    """
        Construct news ID to news INDEX dictionary, index starting from 1
    """
    news_df = pd.read_table(news_path, index_col=None, names=[
                            "newsID", "category", "subcategory", "title", "abstract", "url", "entity_title", "entity_abstract"], quoting=3)

    nid2index = {}
    for v in news_df["newsID"]:
        if v in nid2index:
            continue
        # plus one because all news offsets from 1
        nid2index[v] = len(nid2index) + 1
    save_pickle(nid2index, os.path.join(cache_dir, "nid2index.pkl"))


def construct_uid2index(data_root, cache_root):
    """
        Construct user ID to user IDX dictionary, index starting from 0
    """
    uid2index = {}
    user_df_list = []
    behaviors_file_list = [os.path.join(data_root, "MIND", directory, "behaviors.tsv") for directory in ["MINDlarge_train", "MINDlarge_dev", "MINDlarge_test"]]

    for f in behaviors_file_list:
        user_df_list.append(pd.read_table(f, index_col=None, names=[
                            "imprID", "uid", "time", "hisstory", "abstract", "impression"], quoting=3)["uid"])
    user_df = pd.concat(user_df_list).drop_duplicates()
    for v in user_df:
        uid2index[v] = len(uid2index)
    save_pickle(uid2index, os.path.join(cache_root, "MIND", "uid2index.pkl"))
    return uid2index



class Sequential_Sampler:
    def __init__(self, dataset_length, num_replicas, rank) -> None:
        super().__init__()
        len_per_worker = dataset_length / num_replicas
        self.start = round(len_per_worker * rank)
        self.end = round(len_per_worker * (rank + 1))

    def __iter__(self):
        start = self.start
        end = self.end
        return iter(range(start, end, 1))

    def __len__(self):
        return self.end - self.start
