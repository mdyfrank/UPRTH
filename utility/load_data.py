import sys

import numpy as np
import random as rd
import dgl


class Data(object):
    def __init__(self, path, batch_size, dataset):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        ui_file = path + '/user_item.txt'
        ic_file = path + '/item_category.txt'
        ir_file = path + '/item_rating.txt'
        bT_file = path + '/boughtTogether.txt'
        cpr_file = path + '/comparedTogether.txt'
        ua_file = path + '/user_friend.txt'
        uo_file = path + '/user_groups.txt'

        # get number of users and items
        self.n_users, self.n_items, self.n_cate, self.n_rate = 0, 0, 0, 0
        self.n_age,self.n_job = 0,0
        self.n_train, self.n_test, self.ic_interactions, self.ir_interactions = 0, 0, 0, 0
        self.exist_users = []

        user_item_src, user_item_dst = [],[]
        item_cate_src, item_cate_dst = [],[]
        item_rate_src,item_rate_dst = [],[]
        item_bT_src,item_bT_dst = [],[]
        item_cpr_src,item_cpr_dst = [],[]
        user_age_src,user_age_dst = [],[]
        user_job_src,user_job_dst = [],[]

        self.train_items, self.test_set = {}, {}
        with open(train_file, 'r') as f:
            line = f.readline().strip()
            while line != '':
                line = line.strip('\n').split(' ')
                uid = int(line[0])
                self.exist_users.append(uid)
                items = [int(g) for g in line[1:]]
                self.train_items[uid] = items
                self.n_users = max(self.n_users, uid)
                self.n_items = max(self.n_items, max(items))
                self.n_train += len(items)
                for g in line[1:]:
                    user_item_src.append(uid)
                    user_item_dst.append(int(g))
                line = f.readline().strip()

        with open(test_file, 'r') as f:
            line = f.readline().strip()
            while line != '':
                line = line.strip('\n').split(' ')
                uid = int(line[0])
                items_test = [int(g) for g in line[1:]]
                self.test_set[uid] = items_test
                self.n_items = max(self.n_items, max(items_test))
                self.n_test += len(items_test)
                line = f.readline().strip()
        self.n_users += 1
        self.n_items += 1

        with open(ic_file, 'r') as f:
            line = f.readline().strip()
            while line != '':
                line = line.strip('\n').split(' ')
                self.ic_interactions += 1
                iid = int(line[0])
                if 'xmrec' in dataset:
                    category = int(line[1])
                    self.n_cate = max(self.n_cate, category)
                    item_cate_src.append(iid)
                    item_cate_dst.append(category)
                else:
                    for cate in line[1:]:
                        cate = int(cate)
                        self.n_cate = max(self.n_cate, cate)
                        item_cate_src.append(iid)
                        item_cate_dst.append(cate)
                line = f.readline().strip()
        self.n_cate += 1

        with open(ir_file, 'r') as f:
            line = f.readline().strip()
            while line != '':
                line = line.strip('\n').split(' ')
                self.ir_interactions += 1
                iid = int(line[0])
                if 'xmrec' in dataset:
                    rate = eval(line[1])*2-2
                    item_rate_src.append(iid)
                    item_rate_dst.append(rate)
                else:
                    for rate in line[1:]:
                        rate = int(rate)
                        item_rate_src.append(iid)
                        item_rate_dst.append(rate)
                        self.n_rate = max(self.n_rate, rate)
                line = f.readline().strip()
            if 'xmrec' in dataset:
                self.n_rate = 9
            else:
                self.n_rate += 1

        if 'xmrec' in dataset:
            bT_idx = 0
            with open(bT_file, 'r') as f:
                line = f.readline().strip()
                while line != '':
                    line = line.strip('\n').split(' ')
                    for iid in line:
                        item_bT_src.append(int(iid))
                        item_bT_dst.append(bT_idx)
                    bT_idx += 1
                    line = f.readline().strip()
            cpr_idx = 0
            with open(cpr_file, 'r') as f:
                line = f.readline().strip()
                while line != '':
                    line = line.strip('\n').split(' ')
                    for iid in line:
                        item_cpr_src.append(int(iid))
                        item_cpr_dst.append(cpr_idx)
                    cpr_idx += 1
                    line = f.readline().strip()

            self.n_cluster_bT = bT_idx
            self.n_cluster_cpr = cpr_idx
        elif 'steam' == dataset:
            friend_idx = 0
            with open(ua_file, 'r') as f:
                line = f.readline().strip()
                while line != '':
                    line = line.strip('\n').split(' ')
                    for uid in line:
                        user_age_src.append(int(uid))
                        user_age_dst.append(friend_idx)
                    friend_idx += 1
                    line = f.readline().strip()
            self.n_age = friend_idx

            with open(uo_file, 'r') as f:
                line = f.readline().strip()
                while line != '':
                    line = line.strip('\n').split(' ')
                    # self.ua_interactions += 1
                    uid = int(line[0])
                    groups = line[1:]
                    for g in groups:
                        g = int(g)
                        self.n_job = max(self.n_job, g)
                        user_job_src.append(uid)
                        user_job_dst.append(g)
                    line = f.readline().strip()
            self.n_job += 1

        self.print_statistics(dataset)

        self.item_cate_idx = item_cate_src
        self.item_rate_idx = item_rate_src
        self.cate_label = item_cate_dst
        self.rate_label = item_rate_dst

        if 'xmrec' in dataset:
            data_dict = {
                ('user', 'ui', 'item'): (user_item_src, user_item_dst),
                ('item', 'iu', 'user'): (user_item_dst, user_item_src),
                ('item', 'ic', 'cate'): (item_cate_src, item_cate_dst),
                ('cate', 'ci', 'item'): (item_cate_dst, item_cate_src),
                ('item', 'ir', 'rate'): (item_rate_src, item_rate_dst),
                ('rate', 'ri', 'item'): (item_rate_dst, item_rate_src),
                ('item', 'ib', 'bT_idx'): (item_bT_src, item_bT_dst),
                ('bT_idx', 'bi', 'item'): (item_bT_dst, item_bT_src),
                ('item', 'ip', 'cpr_idx'): (item_cpr_src, item_cpr_dst),
                ('cpr_idx', 'pi', 'item'): (item_cpr_dst, item_cpr_src),
            }
            num_dict = {
                'user': self.n_users, 'item': self.n_items, 'cate': self.n_cate, 'rate': self.n_rate,
                'bT_idx': self.n_cluster_bT,
                'cpr_idx': self.n_cluster_cpr,
            }
            self.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)
        elif 'steam' == dataset:
            data_dict = {
                ('user', 'ui', 'item'): (user_item_src, user_item_dst),
                ('item', 'iu', 'user'): (user_item_dst, user_item_src),
                ('item', 'ic', 'cate'): (item_cate_src, item_cate_dst),
                ('cate', 'ci', 'item'): (item_cate_dst, item_cate_src),
                ('item', 'ir', 'rate'): (item_rate_src, item_rate_dst),
                ('rate', 'ri', 'item'): (item_rate_dst, item_rate_src),
                ('user', 'ua', 'age'): (user_age_src, user_age_dst),
                ('age', 'au', 'user'): (user_age_dst, user_age_src),
                ('user', 'uj', 'job'): (user_job_src, user_job_dst),
                ('job', 'ju', 'user'): (user_job_dst, user_job_src),
            }
            num_dict = {
                'user': self.n_users, 'item': self.n_items, 'cate': self.n_cate, 'rate': self.n_rate,
                'age': self.n_age,
                'job': self.n_job,
            }
            self.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)
            self.age_label = user_age_dst
            self.job_label = user_job_dst
    def print_statistics(self, dataset):
        print('n_users=%d, n_items=%d, n_cate=%d, n_rate=%d' % (
        self.n_users, self.n_items, self.n_cate, self.n_rate))
        if 'xmrec' in dataset:
            print('n_cluster_bT=%d, n_cluster_cpr=%d'%(self.n_cluster_bT, self.n_cluster_cpr))
        elif 'steam' == dataset:
            print('n_ages=%d,n_jobs=%d'%(self.n_age,self.n_job))
        print('n_ui_interactions=%d, n_ic_interactions=%d,n_ir_interactions=%d' % (
            self.n_train + self.n_test, self.ic_interactions, self.ir_interactions))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u] + self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
