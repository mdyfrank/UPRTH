import sys
import collections
import json
split_table = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5, 7: 6, 8: 6, 9: 6, 10: 6, 11: 7
}
path = '../Data/weeplaces'
def initial_txt(path=path, split_table = split_table):
    for _,value in split_table.items():
        with open(path + '/test_%d.txt'%(value), 'w') as f:
            continue
    return

def count_interactions(d):
    count = 0
    length = 0
    for key, value in d.items():
        count += 1
        length += len(value)
    return count, length, length / count
def create_iu_dict(ui_file = '/user_item.txt'):
    ui_file = path + ui_file
    iu_dict = collections.defaultdict(list)
    with open(ui_file,'r') as f:
        line = f.readline().strip()
        while line != '':
            line = line.strip('\n').split(' ')
            uid = int(line[0])
            items = [int(g) for g in line[1:]]
            for i in items:
                iu_dict[i].append(uid)
            line = f.readline().strip()
    return iu_dict

# iu_dict = create_iu_dict()
# print(count_interactions(iu_dict))


def load_train(path = path, split_table = split_table):
    train_file = path + '/train.txt'
    ui_file = path + '/user_item.txt'
    len_dict = collections.defaultdict(int)
    user_split_test = dict([])
    with open(train_file, 'r') as f:
        line = f.readline().strip()
        while line != '':
            line = line.strip('\n').split(' ')
            uid = int(line[0])
            groups = [int(g) for g in line[1:]]
            group_length = len(groups)
            user_split_test[uid] = split_table[min(group_length,max(split_table.keys()))]
            len_dict[group_length] += 1
            line = f.readline().strip()
    print(collections.OrderedDict(sorted(len_dict.items())))
    with open(path + '/user_split_dic', 'w') as outfile:
        json.dump(user_split_test, outfile)
    return user_split_test

def load_test_dict(path = path):
    with open(path + '/user_split_dic') as json_f:
        user_split_test = json.load(json_f)
    return user_split_test

def count_train_edges(path = path, split_table = split_table):
    count_dict = {}
    len_dict = collections.defaultdict(int)
    for _, value in split_table.items():
        count_dict[value] = 0
    train_file = path + '/train.txt'
    with open(train_file, 'r') as f:
        line = f.readline().strip()
        while line != '':
            line = line.strip('\n').split(' ')
            groups = [int(g) for g in line[1:]]
            group_length = len(groups)
            split_idx = split_table[min(group_length, max(split_table.keys()))]
            count_dict[split_idx] += group_length
            len_dict[split_idx] += 1
            line = f.readline().strip()
    print(collections.OrderedDict(sorted(len_dict.items())))
    return count_dict

def split_test(user_split_dict, path = path):
    train_file = path + '/test.txt'
    with open(train_file, 'r') as f:
        line = f.readline().strip()
        while line != '':
            uid = int(line.strip('\n').split(' ')[0])
            user_split_id = user_split_dict[uid]
            with open(path + '/test_%d.txt' % (user_split_id), 'a') as f_new:
                f_new.write(line+'\n')
            line = f.readline().strip()
    return
# print(count_train_edges())
# initial_txt()
# user_split_dict = load_train()
# split_test(user_split_dict, path = path)
# print(load_test_dict())
# split_test(user_split_dict)

# print(user_split_dict)
