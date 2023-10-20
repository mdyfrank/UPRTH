import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run.")
    parser.add_argument('--data_path', nargs='?', default='./Data/',
                        help='Input data path.')
    parser.add_argument('--weights_path', nargs='?', default='./Weights/',
                        help='Input data path.')
    parser.add_argument('--model_name', type=str, default='LightGCN',
                        help='Saved model name.')
    parser.add_argument('--dataset', nargs='?', default='steam',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation. 2 for cn/mx')
    parser.add_argument('--n_cate', type=int, default=12,
                        help='number of category')
    parser.add_argument('--n_rate', type=int, default=9,
                        help='number of rate')
    parser.add_argument('--epoch', type=int, default=10000,
                        help='Number of epoch.')
    parser.add_argument('--user_split',type=int, default=0)

    parser.add_argument('--gat',type=int, default=0)
    parser.add_argument('--sgl',type=int, default=0)
    parser.add_argument('--gcc',type=int, default=0)
    parser.add_argument('--attrimask',type=int, default=1)
    parser.add_argument('--prompt',type=int,default=0)
    parser.add_argument('--train_curve', type=int, default=0)
    parser.add_argument('--pre_gcn',type=int,default=0)
    parser.add_argument('--combine_view',type=int,default=0)
    parser.add_argument('--combine_alpha',type=float,default=0.8)
    parser.add_argument('--hgcn_mix', nargs='?', default='[10, 1e-1]')
    parser.add_argument('--att_conv',type=int,default=-1,help='-1 if no trans_conv')
    parser.add_argument('--show_distance',type=int, default=0)
    parser.add_argument('--loss', type=str, default='align')
    parser.add_argument('--finetune_loss', type=str, default='bpr')

    parser.add_argument('--lightgcn', type=int, default=0,help = '1:Lightgcn, 2:HCCF, 3:DHCF, 4:UltraGCN, 5:HGNN')
    parser.add_argument('--hgcn', type=int, default=1)
    parser.add_argument('--inductive',type=int, default=1)
    parser.add_argument('--induct_ratio',type=float,default=0.5,help='ratio of inductive users')
    parser.add_argument('--pre_train',type=int, default=1)
    parser.add_argument('--classify_as_edge',type=int,default=1,help='only for pretrain')
    parser.add_argument('--multitask_train',type=int, default=0,
                        help='only valid if pre_train = 1')
    parser.add_argument('--pre_train_task', type=int, default=1,
                        help='1 for both & user-item, 2 for rating & user-item, 3 for category & user-item, 4 for user-item only, 5 for cate & rating w/o u-i')
    parser.add_argument('--user_pretrain', type=int, default=1,
                        help='1 for both & all item-pretrain tasks, 2 for group as user-pretrain, 3 for friend as user-pretrain, 0/4 for no user-pretrain')
    parser.add_argument('--item_pretrain',type=int,default=1,
                        help='0 for None, 1 for both (if exist), 2 for boughtTogether only, 3 for comparedTogether only')

    parser.add_argument('--norm_2', type=int, default=-1,
                        help='-0.5 for mx, -1 for cn/mx_C2EP')

    parser.add_argument('--pre_lr', type=float, default=0.01,
                        help='Learning rate. 0.05 for mx/cn')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate. Steam: 0.1(default)/0.05(training curve)')
    parser.add_argument('--regs', nargs='?', default='[0.8, 1e-4]',
                        help='Regularizations.[user-item loss coeffecient, embedding regularization]')

    parser.add_argument('--hgcn_u_hyperedge', type=int, default=1,
                        help='Hypergraph conv on user-group side with user as hyperedge and group as vertex')
    parser.add_argument('--user_hpedge_ig', type=int, default=0,
                        help='Hypergraph conv on user side with user as vertex'
                             '{0: only user-item conv with item as hyperedge;'
                             ' 1: simultaneous user-item and user-group conv;'
                             ' 2: sequential user-item and user-group conv};'
                            '3&4:HGNN and HGNN+')
    parser.add_argument('--random_seed', type=int, default=1,
                        help='1 as defualt')
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='Number of negative samples.')
    parser.add_argument('--contrastive_learning', type=int, default=0)

    parser.add_argument('--ssl_reg', type=float, default=1e-7,
                        help='1e-5 for steam')
    parser.add_argument('--ssl_temp', type=float, default=0.1,
                        help='temperature, 0.1 as default')

    parser.add_argument('--reweight_type', type=int, default=2,
                        help='0: (1-b)/(1-b^x); 1: 1/(bx-b+1); 2: 1/(e^(bx-b)); 3: 1-tanh(bx-b)')
    parser.add_argument('--beta_group', type=float, default=0)
    parser.add_argument('--beta_item', type=float, default=0)

    parser.add_argument('--embed_size', type=int, default=128,
                        help='Embedding size.')
    parser.add_argument('--layer_num', type=int, default=3,
                        help='Output sizes of every layer')

    # parser.add_argument('--batch_size', type=int, default=32768,
                        # help='Batch size.')

    parser.add_argument('--batch_size', type=int, default=16384,
                        help='Batch size.')



    parser.add_argument('--flag_step', type=int, default=5)

    parser.add_argument('--gpu', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--Ks', nargs='?', default='[10,20]',
                        help='Output sizes of every layer')

    parser.add_argument('--fast_test', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='full',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    # parser.add_argument('--mtv_att', type=int, default=0,
    #                     help='0 for no att, 1 for mtv_att')
    # parser.add_argument('--att_type', type=int, default=2,
    #                     help='Att_type')
    # parser.add_argument('--att_update_user', type=int, default=3,
    #                     help='0 for no linear_transform, 1 for linear_transform')
    # parser.add_argument('--res_lambda', type=float, default=0.001,
    #                     help='res_lambda')
    # parser.add_argument('--linear_transform', type=int, default=1,
    #                     help='0 for no linear_transform, 1 for linear_transform')
    # parser.add_argument('--initial_embedding_used', type=int, default=1,
    #                     help='0 for no att, 1 for mtv_att')
    # parser.add_argument('--item_lambda', type=float, default=100,
    #                     help='item_lambda.')
    return parser.parse_args()
