import argparse

from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.utils.tools import load_job_config
from pipeline.runtime.entity import JobParameters
from pipeline.component.homo_nn import HomoNN
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
from pipeline.component import Evaluation



# noinspection PyPep8Naming
class dataset(object):
    breast = {
        "guest": {"name": "breast_homo_guest", "namespace": "experiment"},
        "host": [
            {"name": "breast_homo_host", "namespace": "experiment"},
            {"name": "breast_homo_host", "namespace": "experiment"}
        ]
    }
    vehicle = {
        "guest": {"name": "vehicle_scale_homo_guest", "namespace": "experiment"},
        "host": [
            {"name": "vehicle_scale_homo_host", "namespace": "experiment"},
            {"name": "vehicle_scale_homo_host", "namespace": "experiment"}
        ]
    }
    nsl_kdd = {
        "guest": {"name": "nsl_kdd_train_1", "namespace": "experiment"},
        "host": [
            {"name": "nsl_kdd_train_1", "namespace": "experiment"},
            {"name": "nsl_kdd_train_2", "namespace": "experiment"},
            {"name": "nsl_kdd_train_3", "namespace": "experiment"},
            {"name": "nsl_kdd_train_4", "namespace": "experiment"}
        ],
        "label": "Class"
    }


def main(config="config.yaml", namespace="", num_host=4):
    if isinstance(config, str):
        config = load_job_config(config)
    data = dataset.nsl_kdd
    guest_train_data = data["guest"]
    host_train_data = data["host"][:num_host]
    for d in [guest_train_data, *host_train_data]:
        d["namespace"] = f"{d['namespace']}{namespace}"

    hosts = config.parties.host[:num_host]
    pipeline = PipeLine() \
        .set_initiator(role='guest', party_id=config.parties.guest[0]) \
        .set_roles(guest=config.parties.guest[0], host=hosts, arbiter=config.parties.arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=config.parties.guest[0]).component_param(table=guest_train_data)
    for i in range(num_host):
        reader_0.get_party_instance(role='host', party_id=hosts[i]) \
            .component_param(table=host_train_data[i])

    dataio_0 = DataIO(name="dataio_0", with_label=True)
    dataio_0.get_party_instance(role='guest', party_id=config.parties.guest[0]) \
        .component_param(with_label=True, output_format="dense", label_name='Class')
    dataio_0.get_party_instance(role='host', party_id=hosts).component_param(with_label=True, label_name='Class')

    homo_nn_0 = HomoNN(name="homo_nn_0", encode_label=True, max_iter=15, batch_size=-1,
                       early_stop={"early_stop": "diff", "eps": 0.0001})
    homo_nn_0.add(Dense(units=288, input_shape=(122,), activation="relu"))
    homo_nn_0.add(Dense(units=2, activation="sigmoid"))
    homo_nn_0.compile(optimizer=optimizers.Adam(learning_rate=0.05), metrics=["accuracy"], loss="categorical_crossentropy")

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(homo_nn_0, data=Data(train_data=dataio_0.output.data))
    pipeline.compile()
    job_parameters = JobParameters(backend=config.backend, work_mode=config.work_mode)
    pipeline.fit(job_parameters)
    print(pipeline.get_component("homo_nn_0").get_summary())
    pipeline.deploy_component([dataio_0, homo_nn_0])

    # predict
    predict_pipeline = PipeLine()
    predict_pipeline.add_component(reader_0)
    predict_pipeline.add_component(pipeline, data=Data(predict_input={pipeline.dataio_0.input.data: reader_0.output.data}))
    predict_pipeline.add_component(evaluation_0, data=Data(data=pipeline.homo_nn_0.output.data))
    # run predict model
    predict_pipeline.predict(job_parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str, help="config file")
    parser.add_argument("-num_host", type=int)
    args = parser.parse_args()
    if args.config is not None:
        main(args.config, args.num_host)
    else:
        main()