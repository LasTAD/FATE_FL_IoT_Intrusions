import argparse

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component.hetero_secureboost import HeteroSecureBoost
from pipeline.component.intersection import Intersection
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.component.evaluation import Evaluation
from pipeline.interface.model import Model

from pipeline.utils.tools import load_job_config
from pipeline.runtime.entity import JobParameters
import datasets


def main(config="config.yaml", namespace="", num_host=4):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)

    backend = config.backend
    work_mode = config.work_mode

    parties = config.parties
    guest = parties.guest[0]
    hosts = config.parties.host[:num_host]

    # data sets
    data = datasets.dataset.swat_ait_hetero
    label = data['label']
    guest_train_data = data['train']["guest"]
    host_train_data = data['train']["host"][:num_host]
    for d in [guest_train_data, *host_train_data]:
        d["namespace"] = f"{d['namespace']}{namespace}"

    guest_test_data = data['test']["guest"]
    host_test_data = data['test']["host"][:num_host]
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
    hetero_secure_boost_0 = HeteroSecureBoost(name="hetero_secure_boost_0",
                                              num_trees=30,
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
    pipeline.add_component(hetero_secure_boost_0,
                           data=Data(train_data=intersect_0.output.data, validate_data=intersect_1.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_secure_boost_0.output.data))

    pipeline.compile()
    job_parameters = JobParameters(backend=backend, work_mode=work_mode,
                                   adaptation_parameters={'task_cores_per_node': 2, 'request_task_cores': 2})
    pipeline.fit(job_parameters)
    pipeline.dump("/fate/dip/model/gbdt_model.pkl")

    print("fitting hetero secureboost done, result:")
    print(pipeline.get_component("hetero_secure_boost_0").get_summary())
    # pipeline = PipeLine.load_model_from_file('pipeline_saved.pkl')
    print('start to predict')

    # predict
    # deploy required components
    pipeline.deploy_component([datatr_0, intersect_0, hetero_secure_boost_0, evaluation_0])

    predict_pipeline = PipeLine()
    # add data reader onto predict pipeline
    predict_pipeline.add_component(reader_0)
    # add selected components from train pipeline onto predict pipeline
    # specify data source
    predict_pipeline.add_component(pipeline,
                                   data=Data(predict_input={pipeline.datatr_0.input.data: reader_0.output.data}))

    # run predict model
    predict_pipeline.predict(job_parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
