import argparse

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component.hetero_fast_secureboost import HeteroFastSecureBoost
from pipeline.component.intersection import Intersection
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.component.evaluation import Evaluation
from pipeline.interface.model import Model
from pipeline.utils.tools import JobConfig

from pipeline.utils.tools import load_job_config
from pipeline.runtime.entity import JobParameters

from federatedml.evaluation.metrics import regression_metric, classification_metric
from fate_test.utils import extract_data, parse_summary_result


def main(config="../config.yaml", param="./fate_fast_sbt_param.yaml", namespace="", num_host=4):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)

    if isinstance(param, str):
        param = JobConfig.load_from_file(param)

    backend = config.backend
    work_mode = config.work_mode

    parties = config.parties
    guest = parties.guest[0]
    hosts = config.parties.host[:num_host]

    label = param['label']
    guest_train_data = param['data_guest']['train']
    host_train_data = param['data_host']["train"][:num_host]
    for d in [guest_train_data, *host_train_data]:
        d["namespace"] = f"{d['namespace']}{namespace}"

    guest_test_data = param['data_guest']['test']
    host_test_data = param['data_host']["test"][:num_host]
    for d in [guest_test_data, *host_test_data]:
        d["namespace"] = f"{d['namespace']}{namespace}"

    # init pipeline
    pipeline = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest, host=hosts, )

    # set data reader and data-io

    reader_0, reader_1 = Reader(name="reader_0"), Reader(name="reader_1")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    for i in range(num_host):
        reader_0.get_party_instance(role='host', party_id=hosts[i]) \
            .component_param(table=host_train_data[i])
    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=guest_test_data)
    for i in range(num_host):
        reader_1.get_party_instance(role='host', party_id=hosts[i]) \
            .component_param(table=host_test_data[i])

    datatr_0, datatr_1 = DataTransform(name="datatr_0"), DataTransform(name="datatr_1")
    datatr_0.get_party_instance(role="guest", party_id=guest).component_param(with_label=True, output_format="dense",
                                                                              label_name=label, label_type='int')
    datatr_0.get_party_instance(role="host", party_id=hosts).component_param(with_label=False)
    datatr_1.get_party_instance(role="guest", party_id=guest).component_param(with_label=True, output_format="dense",
                                                                              label_name=label)
    datatr_1.get_party_instance(role="host", party_id=hosts).component_param(with_label=False)

    # data intersect component
    intersect_0 = Intersection(name="intersection_0")
    intersect_1 = Intersection(name="intersection_1")

    # secure boost component
    hetero_fast_secure_boost_0 = HeteroFastSecureBoost(name="hetero_secure_boost_0",
                                                       num_trees=10,
                                                       tree_num_per_party=1,
                                                       task_type="classification",
                                                       objective_param={"objective": "cross_entropy"},
                                                       encrypt_param={"method": "iterativeAffine"},
                                                       tree_param={"max_depth": 3},
                                                       validation_freqs=1)

    # evaluation component
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(datatr_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(datatr_1, data=Data(data=reader_1.output.data), model=Model(datatr_0.output.model))
    pipeline.add_component(intersect_0, data=Data(data=datatr_0.output.data))
    pipeline.add_component(intersect_1, data=Data(data=datatr_1.output.data))
    pipeline.add_component(hetero_fast_secure_boost_0,
                           data=Data(train_data=intersect_0.output.data, validate_data=intersect_1.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_fast_secure_boost_0.output.data))

    pipeline.compile()
    job_parameters = JobParameters(backend=backend, work_mode=work_mode,
                                   adaptation_parameters={'task_cores_per_node': 2, 'request_task_cores': 2})
    pipeline.fit(job_parameters)

    sbt_0_data = pipeline.get_component("hetero_fast_sbt_0").get_output_data().get("data")
    sbt_1_data = pipeline.get_component("hetero_fast_sbt_1").get_output_data().get("data")
    sbt_0_score = extract_data(sbt_0_data, "predict_result")
    sbt_0_label = extract_data(sbt_0_data, "label")
    sbt_1_score = extract_data(sbt_1_data, "predict_result")
    sbt_1_label = extract_data(sbt_1_data, "label")
    sbt_0_score_label = extract_data(sbt_0_data, "predict_result", keep_id=True)
    sbt_1_score_label = extract_data(sbt_1_data, "predict_result", keep_id=True)
    metric_summary = parse_summary_result(pipeline.get_component("evaluation_0").get_summary())
    metric_sbt = {
        "score_diversity_ratio": classification_metric.Distribution.compute(sbt_0_score_label, sbt_1_score_label),
        "ks_2samp": classification_metric.KSTest.compute(sbt_0_score, sbt_1_score),
        "mAP_D_value": classification_metric.AveragePrecisionScore().compute(sbt_0_score, sbt_1_score, sbt_0_label,
                                                                             sbt_1_label)}
    metric_summary["distribution_metrics"] = {"hetero_fast_sbt": metric_sbt}
    data_summary = {"train": {"guest": guest_train_data["name"], "host": host_train_data["name"]},
                    "test": {"guest": guest_train_data["name"], "host": host_train_data["name"]}
                    }
    return data_summary, metric_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
