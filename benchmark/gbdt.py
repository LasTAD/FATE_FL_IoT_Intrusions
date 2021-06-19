import argparse
import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score, accuracy_score


def main(config="../config.yaml", param="gbdt_conf.yaml"):

    # if isinstance(param, str):
    #     param = JobConfig.load_from_file(param)
    data = '../Datasets/SWaT/Physical/to_FATE/hetero/swat_phys_train_p6_label.csv'
    label_name = 'Normal/Attack'
    # data = param["data"]
    # label_name = param["label_name"]

    # print('config is {}'.format(config))
    # if isinstance(config, str):
    #     config = JobConfig.load_from_file(config)
    #     data_base_dir = config["data_base_dir"]
    #     print('data base dir is', data_base_dir)
    # else:
    #     data_base_dir = config.data_base_dir

    # prepare data
    df = pd.read_csv(os.path.join(data))
    y = df[label_name]
    X = df.drop(label_name, axis=1)
    clf = GradientBoostingClassifier(random_state=0, n_estimators=120 if 'epsilon' in data else 1)
    clf.fit(X, y)
    y_prob = clf.predict(X)

    try:
        auc_score = roc_auc_score(y, y_prob)
        acc_score = accuracy_score(y, y_prob)
    except:
        print(f"no auc score available")
        return

    result = {"auc": auc_score,
              "acc": acc_score}
    print(result)
    return {}, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY SKLEARN JOB")
    parser.add_argument("-param", type=str,
                        help="config file for params")
    args = parser.parse_args()
    if args.param is not None:
        main(args.param)
    main()