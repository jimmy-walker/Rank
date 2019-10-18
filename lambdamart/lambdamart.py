import os
import lightgbm as lgb
from sklearn import datasets as ds
import pandas as pd

import numpy as np
from datetime import datetime
import sys
from sklearn.preprocessing import OneHotEncoder

def split_data_from_keyword(data_read, data_group, data_feats):
    '''
    利用pandas
    转为lightgbm需要的格式进行保存
    :param data_read:
    :param data_save:
    :return:
    '''
    with open(data_group, 'w', encoding='utf-8') as group_path:
        with open(data_feats, 'w', encoding='utf-8') as feats_path:
            dataframe = pd.read_csv(data_read,
                                    sep=' ',
                                    header=None,
                                    encoding="utf-8",
                                    engine='python')
            current_keyword = ''
            current_data = []
            group_size = 0
            for _, row in dataframe.iterrows():
                feats_line = [str(row[0])]
                for i in range(2, len(dataframe.columns) - 1):
                    feats_line.append(str(row[i]))
                if current_keyword == '':
                    current_keyword = row[1]
                if row[1] == current_keyword:
                    current_data.append(feats_line)
                    group_size += 1
                else:
                    for line in current_data:
                        feats_path.write(' '.join(line))
                        feats_path.write('\n')
                    group_path.write(str(group_size) + '\n')

                    group_size = 1
                    current_data = []
                    current_keyword = row[1]
                    current_data.append(feats_line)

            for line in current_data:
                feats_path.write(' '.join(line))
                feats_path.write('\n')
            group_path.write(str(group_size) + '\n')

def save_data(group_data, output_feature, output_group):
    '''
    group与features分别进行保存
    :param group_data:
    :param output_feature:
    :param output_group:
    :return:
    '''
    if len(group_data) == 0:
        return
    output_group.write(str(len(group_data)) + '\n')
    for data in group_data:
        # 只包含非零特征
        # feats = [p for p in data[2:] if float(p.split(":")[1]) != 0.0]
        feats = [p for p in data[2:]]
        output_feature.write(data[0] + ' ' + ' '.join(feats) + '\n') # data[0] => level ; data[2:] => feats

def process_data_format(test_path, test_feats, test_group):
    '''
     转为lightgbm需要的格式进行保存
     '''
    with open(test_path, 'r', encoding='utf-8') as fi:
        with open(test_feats, 'w', encoding='utf-8') as output_feature:
            with open(test_group, 'w', encoding='utf-8') as output_group:
                group_data = []
                group = ''
                for line in fi:
                    if not line:
                        break
                    if '#' in line:
                        line = line[:line.index('#')]
                    splits = line.strip().split()
                    if splits[1] != group: # qid => splits[1]
                        save_data(group_data, output_feature, output_group)
                        group_data = []
                    group = splits[1]
                    group_data.append(splits)
                save_data(group_data, output_feature, output_group)

def load_data(feats, group):
    '''
    加载数据
    分别加载feature,label,query
    '''
    x_train, y_train = ds.load_svmlight_file(feats)
    q_train = np.loadtxt(group)
    return x_train, y_train, q_train

def load_data_from_raw(raw_data):
    with open(raw_data, 'r', encoding='utf-8') as testfile:
        test_X, test_y, test_qids, comments = letor.read_dataset(testfile)
    return test_X, test_y, test_qids, comments

def train(x_train, y_train, q_train, model_save_path):
    '''
    模型的训练和保存
    '''
    train_data = lgb.Dataset(x_train, label=y_train, group=q_train)
    params = {
        'task': 'train',  # 执行的任务类型
        'boosting_type': 'gbrt',  # 基学习器
        'objective': 'lambdarank',  # 排序任务(目标函数)
        'metric': 'ndcg',  # 度量的指标(评估函数)
        'max_position': 10,  # @NDCG 位置优化
        'metric_freq': 1,  # 每隔多少次输出一次度量结果
        'train_metric': True,  # 训练时就输出度量结果
        'ndcg_at': [10],
        'max_bin': 255,  # 一个整数，表示最大的桶的数量。默认值为 255。lightgbm 会根据它来自动压缩内存。如max_bin=255 时，则lightgbm 将使用uint8 来表示特征的每一个值。
        'num_iterations': 500,  # 迭代次数
        'learning_rate': 0.01,  # 学习率
        'num_leaves': 31,  # 叶子数
        # 'max_depth':6,
        'tree_learner': 'serial',  # 用于并行学习，‘serial’： 单台机器的tree learner
        'min_data_in_leaf': 30,  # 一个叶子节点上包含的最少样本数量
        'verbose': 2  # 显示训练时的信息
    }
    gbm = lgb.train(params, train_data, valid_sets=[train_data])
    gbm.save_model(model_save_path)

def predict(x_test, comments, model_input_path):
    '''
    预测得分并排序
    '''
    gbm = lgb.Booster(model_file=model_input_path)  # 加载model

    ypred = gbm.predict(x_test)

    predicted_sorted_indexes = np.argsort(ypred)[::-1]  # 返回从大到小的索引

    t_results = comments[predicted_sorted_indexes]  # 返回对应的comments,从大到小的排序

    return t_results

def test_data_ndcg(model_path, test_path):
    '''
    评估测试数据的ndcg
    '''
    with open(test_path, 'r', encoding='utf-8') as testfile:
        test_X, test_y, test_qids, comments = letor.read_dataset(testfile)

    gbm = lgb.Booster(model_file=model_path)
    test_predict = gbm.predict(test_X)

    average_ndcg, _ = ndcg.validate(test_qids, test_y, test_predict, 60)
    # 所有qid的平均ndcg
    print("all qid average ndcg: ", average_ndcg)
    print("job done!")

def plot_print_feature_importance(model_path):
    '''
    打印特征的重要度
    '''
    #模型中的特征是Column_数字,这里打印重要度时可以映射到真实的特征名
    feats_dict = {
        'Column_0': '特征0名称',
        'Column_1': '特征1名称',
        'Column_2': '特征2名称',
        'Column_3': '特征3名称',
        'Column_4': '特征4名称',
        'Column_5': '特征5名称',
        'Column_6': '特征6名称',
        'Column_7': '特征7名称',
        'Column_8': '特征8名称',
        'Column_9': '特征9名称',
        'Column_10': '特征10名称',
    }
    if not os.path.exists(model_path):
        print("file no exists! {}".format(model_path))
        sys.exit(0)

    gbm = lgb.Booster(model_file=model_path)

    # 打印和保存特征重要度
    importances = gbm.feature_importance(importance_type='split')
    feature_names = gbm.feature_name()

    sum = 0.
    for value in importances:
        sum += value

    for feature_name, importance in zip(feature_names, importances):
        if importance != 0:
            feat_id = int(feature_name.split('_')[1]) + 1
            print('{} : {} : {} : {}'.format(feat_id, feats_dict[feature_name], importance, importance / sum))

def get_leaf_index(data, model_path):
    '''
    得到叶结点并进行one-hot编码
    '''
    gbm = lgb.Booster(model_file=model_path)
    ypred = gbm.predict(data, pred_leaf=True)

    one_hot_encoder = OneHotEncoder()
    x_one_hot = one_hot_encoder.fit_transform(ypred)
    print(x_one_hot.toarray()[0])

if __name__ == '__main__':
    model_path = "保存模型的路径"

    if len(sys.argv) != 2:
        print("Usage: python main.py [-process | -train | -predict | -ndcg | -feature | -leaf]")
        sys.exit(0)

    if sys.argv[1] == '-process':
        # 训练样本的格式与ranklib中的训练样本是一样的,但是这里需要处理成lightgbm中排序所需的格式
        # lightgbm中是将样本特征和group分开保存为txt的,什么意思呢,看下面解释
        '''
        feats:
        1 1:0.2 2:0.4 ...
        2 1:0.2 2:0.4 ...
        1 1:0.2 2:0.4 ...
        3 1:0.2 2:0.4 ...
        group:
        2
        4
        这里group中2表示前2个是一个qid,4表示后两个是一个qid
        '''
        raw_data_path = '训练样本集路径'
        data_feats = '特征保存路径'
        data_group = 'group保存路径'
        process_data_format(raw_data_path, data_feats, data_group)

    elif sys.argv[1] == '-train':
        # train
        train_start = datetime.now()
        data_feats = '特征保存路径'
        data_group = 'group保存路径'
        x_train, y_train, q_train = load_data(data_feats, data_group)
        train(x_train, y_train, q_train, model_path)
        train_end = datetime.now()
        consume_time = (train_end - train_start).seconds
        print("consume time : {}".format(consume_time))

    elif sys.argv[1] == '-predict':
        train_start = datetime.now()
        raw_data_path = '需要预测的数据路径'#格式如ranklib中的数据格式
        test_X, test_y, test_qids, comments = load_data_from_raw(raw_data_path)
        t_results = predict(test_X, comments, model_path)
        train_end = datetime.now()
        consume_time = (train_end - train_start).seconds
        print("consume time : {}".format(consume_time))

    elif sys.argv[1] == '-ndcg':
        # ndcg
        test_path = '测试的数据路径'#评估测试数据的平均ndcg
        test_data_ndcg(model_path, test_path)

    elif sys.argv[1] == '-feature':
        plot_print_feature_importance(model_path)

    elif sys.argv[1] == '-leaf':
        #利用模型得到样本叶结点的one-hot表示
        raw_data = '测试数据路径'#
        with open(raw_data, 'r', encoding='utf-8') as testfile:
            test_X, test_y, test_qids, comments = letor.read_dataset(testfile)
        get_leaf_index(test_X, model_path)