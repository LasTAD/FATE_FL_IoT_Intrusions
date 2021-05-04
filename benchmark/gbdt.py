import argparse
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from pipeline.utils.tools import JobConfig


def main(config="../config.yaml", param="gbdt_conf.yaml"):

    if isinstance(param, str):
        param = JobConfig.load_from_file(param)

    data = param["data"]
    label_name = param["label_name"]

    print('config is {}'.format(config))
    if isinstance(config, str):
        config = JobConfig.load_from_file(config)
        data_base_dir = config["data_base_dir"]
        print('data base dir is', data_base_dir)
    else:
        data_base_dir = config.data_base_dir

    # prepare data
    df = pd.read_csv(os.path.join(data_base_dir, data))
    y = df[label_name]
    X = df.drop(label_name, axis=1)
    clf = GradientBoostingClassifier(random_state=0, n_estimators=120 if 'epsilon' in data else 10)
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