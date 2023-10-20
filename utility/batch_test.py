import sys

import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import heapq
import torch
import numpy as np
from evaluator import eval_score_matrix_foldout
from utility.split_dataset import load_test_dict

cores = multiprocessing.cpu_count()
print(cores)
args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size, dataset = args.dataset)
USER_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
# user_test_dict = load_test_dict(path=args.data_path + args.dataset)
# print(user_test_dict)
# sys.exit()
BATCH_SIZE = args.batch_size


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    # print(K_max_item_score)
    # sys.exit(0)
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    # print(r)
    # print(metrics.recall_at_k(r, 10, len(user_pos_test)))
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def get_performance_recall_ndcg(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))

    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg)}


def test_one_user(x):
    # user u's ratings for items
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    # try:
    training_items = data_generator.train_items[u]
    # except Exception:
    #     training_items = []
    # user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
    # print(r)
    # print(get_performance_recall_ndcg(user_pos_test, r, auc, Ks))
    # sys.exit()
    return get_performance_recall_ndcg(user_pos_test, r, auc, Ks)

def test_one_user_uid(x):
    # user u's ratings for items
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    # try:
    training_items = data_generator.train_items[u]
    # except Exception:
    #     training_items = []
    # user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
    return (u,get_performance_recall_ndcg(user_pos_test, r, auc, Ks))

def test(users_to_test, embedding_h, user_split=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    recall_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    count_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    ndcg_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    pool = multiprocessing.Pool(cores)

    u_batch_size = args.batch_size

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1 if n_test_users % u_batch_size != 0 else n_test_users // u_batch_size

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        print('test batch %d' % u_batch_id)
        test_users_embedding = embedding_h['user'][user_batch]
        full_items_embedding = embedding_h['item']
        rate_batch = torch.matmul(test_users_embedding, full_items_embedding.t()).detach().cpu()

        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        if user_split:
            print('no split')
            sys.exit(0)
            # uid_batch_result = pool.map(test_one_user_uid, user_batch_rating_uid)
            # count += len(uid_batch_result)
            # for re in uid_batch_result:
            #     # result['precision'] += re['precision'] / n_test_users
            #     user_part = user_test_dict[str(re[0])]
            #     recall = re[1]['recall']
            #     ndcg = re[1]['ndcg']
            #     result['recall'] += recall / n_test_users
            #     result['ndcg'] += ndcg / n_test_users
            #     count_dict[user_part] += 1
            #     recall_dict[user_part] += recall
            #     ndcg_dict[user_part] += ndcg
        else:
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                # result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                # result['hit_ratio'] += re['hit_ratio'] / n_test_users
                # result['auc'] += re['auc'] / n_test_users
    assert count == n_test_users
    pool.close()
    if user_split:
        print(count_dict)
        for key in recall_dict.keys():
            recall_dict[key] = recall_dict[key] / count_dict[key]

        for key in ndcg_dict.keys():
            ndcg_dict[key] = ndcg_dict[key] / count_dict[key]

    return result, recall_dict, ndcg_dict

def test_cpp(users_to_test, embedding_h, device = None):
    top_show = np.sort(eval(args.Ks))
    max_top = max(top_show)
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    u_batch_size = args.batch_size

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1 if n_test_users % u_batch_size != 0 else n_test_users // u_batch_size

    count = 0
    all_result = []
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start: end]
        print('test batch %d' % u_batch_id)
        test_users_embedding = embedding_h['user'][user_batch]
        full_items_embedding = embedding_h['item']
        rate_batch = torch.matmul(test_users_embedding, full_items_embedding.t()).detach().cpu()
        test_items = []
        for user in user_batch:
            test_items.append(data_generator.test_set[user])  # (B, #test_items)

        # set the ranking scores of training items to -inf,
        # then the training items will be sorted at the end of the ranking list.
        for idx, user in enumerate(user_batch):
            train_items_off = data_generator.train_items[user]
            rate_batch[idx][train_items_off] = -np.inf
        batch_result = eval_score_matrix_foldout(rate_batch, test_items, max_top)  # (B,k*metric_num), max_top= 20
        # print(batch_result.shape)
        # sys.exit()
        count += len(batch_result)
        all_result.append(batch_result)
        # print(all_result)
    assert count == n_test_users
    all_result = np.concatenate(all_result, axis=0)
    final_result = np.mean(all_result, axis=0)  # mean

    final_result = np.reshape(final_result, newshape=[5, max_top])
    final_result = final_result[:, top_show - 1]
    final_result = np.reshape(final_result, newshape=[5, len(top_show)])
    # result['precision'] += final_result[0]
    result['recall'] += final_result[1]
    result['ndcg'] += final_result[3]
    return result, None, None