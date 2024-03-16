import sentencepiece as spm
import fire
import json
import os
import os.path as path
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from mytools.tool_utils import FileUtils, HFUtils, FaissUtils
from tempfile import mkdtemp
import faiss
from faiss.contrib.ondisk import merge_ondisk
import numpy as np
import random
random.seed(10086)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ],
)


def kmeans(dataprefix, index_dir, num_data_shards, kmeans_on_gpu=False, num_train=200000, num_centroids=4096, normalize=False, log=False):
    assert index_dir
    FileUtils.check_dirs(index_dir)
    niter = 20
    idx_data_file_list = ["{}.{}.repr.idx".format(dataprefix, i) for i in range(num_data_shards)]
    repr_data_file_list = ["{}.{}.repr.dat".format(dataprefix, i) for i in range(num_data_shards)]
    xb = []
    logging.info("Loading data...")
    for idx_file, repr_file in zip(idx_data_file_list, repr_data_file_list):
        sub_xb, _ = HFUtils.load_extracted_repr(idx_file, repr_file, normalize, log=log)
        topk = sub_xb.shape[0]
        logging.info("Loaded {} vectors from {}".format(topk, path.basename(repr_file)))
        xb.append(sub_xb)
    xb = np.concatenate(xb, axis=0)
    logging.info("Shape of xb: {}".format(xb.shape))
    n, d = xb.shape
    train_ids = random.sample(range(n), k=num_train)
    xb = xb[train_ids]
    logging.info("Training kmeans with {} centroids and {} vectors...".format(num_centroids, num_train))
    kmeans = faiss.Kmeans(d, num_centroids, niter=niter, verbose=True, gpu=kmeans_on_gpu)
    kmeans.train(xb)
    clst_path = path.join(index_dir, 'centroids.kmeans.{}.dat'.format(num_centroids))
    logging.info("Saving centroids after runing Kmeans...")
    torch.save(torch.from_numpy(kmeans.centroids), clst_path)


def group_by_cluster(dataprefix, centroids, num_data_shards, normalize=False, log=False):
    logging.info("Loading centroids...")
    centroids = torch.load(centroids)
    nc, d = centroids.size()
    kmeans_index = faiss.IndexFlatL2(d)
    kmeans_index.add(centroids.numpy())
    if torch.cuda.is_available():
        kmeans_index = FaissUtils.to_single_gpu(kmeans_index, useFloat16=False)
    grouped_xid = [[] for _ in range(nc)]
    idx_data_file_list = ["{}.{}.repr.idx".format(dataprefix, i) for i in range(num_data_shards)]
    repr_data_file_list = ["{}.{}.repr.dat".format(dataprefix, i) for i in range(num_data_shards)]
    for idx_file, repr_file in zip(idx_data_file_list, repr_data_file_list):
        x, xid = HFUtils.load_extracted_repr(idx_file, repr_file, normalize, log=False)
        logging.info("Loading {} vectors from {}".format(len(x), repr_file))
        D, CI = kmeans_index.search(x, 1)
        for i, c in zip(xid, CI):
            grouped_xid[c[0]].append(i)
    logging.info("{} cases are classified into {} groups".format(sum([len(it) for it in grouped_xid]), len(grouped_xid)))
    torch.save(grouped_xid, "{}.group.idx".format(dataprefix))


def train_index(dataprefix, index_dir, num_data_shards=-1, index_type="IVF1024,PQ64", index_on_gpu=False, num_train=200000, store_centroids=False, normalize=False, log=False, enable_memmap=False, overwrite_memmap=False, cache_dir=""):
    """
    There is no need to train for L2 and IP search. In other words, only quantization-based
    index needs to be trained before searching.
    """
    assert index_dir
    if isinstance(index_type, str):
        prefix = ("-".join(index_type.split(","))).lower()
    elif isinstance(index_type, tuple):
        prefix = ("-".join(index_type)).lower()
        index_type = ",".join(index_type)
    FileUtils.check_dirs(index_dir)
    if num_data_shards < 0:
        from glob import glob
        num_data_shards = len(list(glob("{}.*.repr.idx".format(dataprefix))))
        logging.info("Found {} data shards".format(num_data_shards))
    idx_data_file_list = ["{}.{}.repr.idx".format(dataprefix, i) for i in range(num_data_shards)]
    repr_data_file_list = ["{}.{}.repr.dat".format(dataprefix, i) for i in range(num_data_shards)]
    logging.info("Loading data...")
    if enable_memmap:
        cache_dir = cache_dir if cache_dir else index_dir
        FileUtils.check_dirs(cache_dir)
        dat_file = "{}/{}.memmap.dat".format(cache_dir, prefix)
        dat_file_exists = FileUtils.exists(dat_file)
        num_x = 0
        for idx_file in idx_data_file_list:
            num_x += len(FileUtils.load_file(idx_file))
        first_xb, _ = HFUtils.load_extracted_repr(idx_data_file_list[0], repr_data_file_list[0], normalize, log=log)
        d = first_xb.shape[1]
        xb = np.memmap(dat_file, dtype='float32', mode='w+', shape=(num_x, d))
        s = 0
        if not dat_file_exists or overwrite_memmap:
            logging.info("No memmap file found. Writing data to new memmap file")
            for idx_file, repr_file in zip(idx_data_file_list, repr_data_file_list):
                sub_xb, _ = HFUtils.load_extracted_repr(idx_file, repr_file, normalize, log=log)
                sub_xb_size = sub_xb.shape[0]
                logging.info("Loaded {} vectors from {}".format(sub_xb_size, path.basename(repr_file)))
                e = s + sub_xb_size
                xb[s:e] = sub_xb
                s = e
        else:
            logging.info("Reuse existing memmap file from {}".format(dat_file))
    else:
        xb = []
        for idx_file, repr_file in zip(idx_data_file_list, repr_data_file_list):
            sub_xb, _ = HFUtils.load_extracted_repr(idx_file, repr_file, normalize, log=log)
            sub_xb_size = sub_xb.shape[0]
            logging.info("Loaded {} vectors from {}".format(sub_xb_size, path.basename(repr_file)))
            xb.append(sub_xb)
        xb = np.concatenate(xb, axis=0)
    logging.info("Shape of xb: {}".format(xb.shape))
    n, d = xb.shape
    train_ids = random.sample(range(n), k=num_train)
    xb = xb[train_ids]

    logging.info("{} vectors with {} dimensions are sampled...".format(num_train, d))
    index = faiss.index_factory(d, index_type)
    if index_on_gpu:
        index = FaissUtils.to_single_gpu(index, useFloat16=False)
    logging.info("Training index...")
    index.train(xb)
    logging.info("Saving index...")
    save_file = path.join(index_dir, prefix + ".trained.index")
    if index_on_gpu:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, save_file)
    if store_centroids:
        clst_path = path.join(index_dir, 'centroids.{}.dat'.format(prefix))
        logging.info("Saving centroids after runing Kmeans...")
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        torch.save(torch.from_numpy(centroids), clst_path)


def build_index(dataprefix, index_dir, num_data_shards=-1, index_type="IVF1024,PQ64", index_on_gpu=False, num_index_shards=1, normalize=False, log=False, debug=False, enable_memmap=False, overwrite_memmap=False, cache_dir="", index_name_tag=""):
    assert index_dir
    if isinstance(index_type, str):
        prefix = ("-".join(index_type.split(","))).lower()
    elif isinstance(index_type, tuple):
        prefix = ("-".join(index_type)).lower()
    FileUtils.check_dirs(index_dir)
    if num_data_shards < 0:
        from glob import glob
        num_data_shards = len(list(glob("{}.*.repr.idx".format(dataprefix))))
        logging.info("Found {} data shards".format(num_data_shards))
    idx_data_file_list = ["{}.{}.repr.idx".format(dataprefix, i) for i in range(num_data_shards)]
    repr_data_file_list = ["{}.{}.repr.dat".format(dataprefix, i) for i in range(num_data_shards)]
    logging.info("Loading data...")
    if enable_memmap:
        cache_dir = cache_dir if cache_dir else index_dir
        FileUtils.check_dirs(cache_dir)
        if index_name_tag:
            dat_file = "{}/{}.{}.memmap.dat".format(cache_dir, prefix, index_name_tag)
        else:
            dat_file = "{}/{}.memmap.dat".format(cache_dir, prefix)
        dat_file_exists = FileUtils.exists(dat_file)
        logging.info("Enabling memmap under path {} ...".format(dat_file))
        xids = []
        for idx_file in idx_data_file_list:
            xids.append(np.array(FileUtils.load_file(idx_file)))
        xids = np.concatenate(xids, axis=0)
        num_x = xids.shape[0]
        first_xb, _ = HFUtils.load_extracted_repr(idx_data_file_list[0], repr_data_file_list[0], normalize, log=log)
        d = first_xb.shape[1]
        xb = np.memmap(dat_file, dtype='float32', mode='w+', shape=(num_x, d))
        if not dat_file_exists or overwrite_memmap:
            logging.info("No memmap file found. Writing data to new memmap file")
            s = 0
            for idx_file, repr_file in zip(idx_data_file_list, repr_data_file_list):
                sub_xb, _ = HFUtils.load_extracted_repr(idx_file, repr_file, normalize, log=log)
                logging.info("Loaded {} vectors from {}".format(sub_xb.shape[0], path.basename(repr_file)))
                sub_sb_size = sub_xb.shape[0]
                e = s + sub_sb_size
                xb[s:e] = sub_xb
                logging.info("Written to xb[{}:{}]".format(s, e))
                s = e
        else:
            logging.info("Reuse existing memmap file from {}".format(dat_file))
    else:
        xb, xids = [], []
        loaded_num = 0
        for idx_file, repr_file in zip(idx_data_file_list, repr_data_file_list):
            sub_xb, sub_xids = HFUtils.load_extracted_repr(idx_file, repr_file, normalize, log=log)
            logging.info("Loaded {} vectors from {}".format(sub_xb.shape[0], path.basename(repr_file)))
            xb.append(sub_xb)
            xids.append(sub_xids)
            loaded_num += 1
            if debug and loaded_num >= 5:
                break
        xb = np.concatenate(xb, axis=0)
        xids = np.concatenate(xids, axis=0)
    logging.info("Shape of xb: {}".format(xb.shape))
    logging.info("Shape of xids: {}".format(xids.shape))
    n, d = xb.shape
    index_shard_size = n // num_index_shards
    if (n % num_index_shards) != 0:
        index_shard_size += 1

    if prefix == "flatl2":
        logging.info("Initializing FlatL2...")
        index = faiss.IndexIDMap(faiss.IndexFlatL2(d))
    elif prefix == "flatip":
        logging.info("Initializing FlatIP...")
        index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
    else:
        empty_index_file = path.join(index_dir,  prefix + ".trained.index")
        logging.info("Loading index file from: {}".format(empty_index_file))
        index = faiss.read_index(empty_index_file)

    block_fnames = []
    for index_shard_idx in range(num_index_shards):
        s, e = index_shard_idx * index_shard_size, (index_shard_idx+1) * index_shard_size
        sub_xb, sub_ids = xb[s:e], xids[s:e]
        index.add_with_ids(sub_xb, sub_ids)
        if num_index_shards > 1:
            if index_name_tag:
                shard_file = path.join(index_dir, prefix + ".{}".format(index_name_tag) + ".shard.{}.index".format(index_shard_idx))
            else:
                shard_file = path.join(index_dir, prefix + ".shard.{}.index".format(index_shard_idx))
        else:
            if index_name_tag:
                shard_file = path.join(index_dir, prefix + ".{}".format(index_name_tag) + ".index")
            else:
                shard_file = path.join(index_dir, prefix + ".index")
        block_fnames.append(shard_file)
        if index_on_gpu:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, shard_file)
        logging.info("{} vectors have been saved to {}".format(sub_xb.shape[0], path.basename(shard_file)))
        if prefix == "flatl2":
            index = faiss.IndexIDMap(faiss.IndexFlatL2(d))
        elif prefix == "flatip":
            index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
        else:
            index = faiss.read_index(empty_index_file)

    if prefix != "flatl2" and prefix != "flatip" and num_index_shards > 1:
        index = faiss.read_index(empty_index_file)
        logging.info("Merging index")
        if index_name_tag:
            merge_ondisk(index, block_fnames, path.join(index_dir, prefix + ".{}".format(index_name_tag) + '.merged.ivfdata'))
            faiss.write_index(index, path.join(index_dir, prefix + ".{}".format(index_name_tag) + '.merged.index'))
        else:
            merge_ondisk(index, block_fnames, path.join(index_dir, prefix + '.merged.ivfdata'))
            faiss.write_index(index, path.join(index_dir, prefix + '.merged.index'))            


def search_single_file(queryprefix, index_path, savepath, index_on_gpu=False, search_batch_size=64, search_topk=32, nprobe=64, normalize=False, log=False, gpu_num=1):
    FileUtils.check_dirs(FileUtils.get_dir(savepath))
    logging.info("Loading index...")
    idx_file = "{}.repr.idx".format(queryprefix)
    repr_file = "{}.repr.dat".format(queryprefix)
    index = faiss.read_index(index_path)
    index.nprobe = nprobe

    if index_on_gpu:
        if gpu_num > 1:
            device_ids = range(gpu_num)
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            resources = [faiss.StandardGpuResources() for _ in device_ids]
            vres = faiss.GpuResourcesVector()
            vdev = faiss.Int32Vector()
            for i, res in zip(device_ids, resources):
                vdev.push_back(i)
                vres.push_back(res)
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
        else:
            index = FaissUtils.to_single_gpu(index, useFloat16=False)
    if index_on_gpu:
        logging.info("Searching on {} gpu...".format(gpu_num))
    else:
        logging.info("Searching on cpu...")
    xq, xq_ids = HFUtils.load_extracted_repr(idx_file, repr_file, normalize, log=log)
    D, I = FaissUtils.search(xq, index, search_batch_size, search_topk)
    results = {
        "query_ids": xq_ids, "search_ids": I.tolist(), "search_dist": D.tolist()
    }
    FileUtils.save_to_disk(results, savepath, 'pt')


def search_files(queryprefix_list, index_path, save_suffix, index_on_gpu=False, search_batch_size=64, search_topk=32, nprobe=64, normalize=False, log=False, gpu_num=1):
    logging.info("Loading index...")
    index = faiss.read_index(index_path)
    index.nprobe = nprobe
    if index_on_gpu:
        if gpu_num > 1:
            device_ids = range(gpu_num)
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True 
            resources = [faiss.StandardGpuResources() for _ in device_ids]
            vres = faiss.GpuResourcesVector()
            vdev = faiss.Int32Vector()
            for i, res in zip(device_ids, resources):
                vdev.push_back(i)
                vres.push_back(res)
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
        else:
            index = FaissUtils.to_single_gpu(index, useFloat16=False)
    if index_on_gpu:
        logging.info("Searching on {} gpu...".format(gpu_num))
    else:
        logging.info("Searching on cpu...")
    for queryprefix in queryprefix_list:
        idx_file = "{}.repr.idx".format(queryprefix)
        repr_file = "{}.repr.dat".format(queryprefix)
        xq, xq_ids = HFUtils.load_extracted_repr(idx_file, repr_file, normalize, log=log)
        D, I = FaissUtils.search(xq, index, search_batch_size, search_topk)
        results = {
            "query_ids": xq_ids, "search_ids": I.tolist(), "search_dist": D.tolist()
        }
        savepath = "{}.{}".format(queryprefix, save_suffix)
        FileUtils.check_dirs(FileUtils.get_dir(savepath))
        FileUtils.save_to_disk(results, savepath, 'pt')


def map_retrieval_to_data(datastore_file_list, retrieval_output_path, save_tag, topk=1000, remove_dup_sent=False, min_dist=0):
    logging.info("Loading index...")
    retrieval_output = FileUtils.load_file(retrieval_output_path)
    indices, scores = [], []
    for sids, dists in zip(retrieval_output['search_ids'], retrieval_output['search_dist']):
        indices.extend(sids[:topk])
        scores.extend(dists[:topk])
    for ds_file in datastore_file_list.split(","):
        if not ds_file:
            continue
        data = FileUtils.load_file(ds_file)
        # sampled_data = ["{}\t{:.4f}".format(data[i], s) for i, s in zip(indices, scores)]
        sampled_data = []
        visited_sents = set()
        for i, s in zip(indices, scores):
            sent = data[i]
            if s < min_dist or (remove_dup_sent and sent in visited_sents):
                continue
            visited_sents.add(sent)
            sampled_data.append(sent)
        FileUtils.save_file(sampled_data, FileUtils.handle_file_extension(ds_file, save_tag))


def main():
    fire.Fire({
        "kmeans": kmeans,
        "group_by_cluster": group_by_cluster,
        "search_single_file": search_single_file,
        "build_index": build_index,
        "train_index": train_index,
        "map_retrieval_to_data": map_retrieval_to_data
    })

if __name__ == "__main__":
    main()
