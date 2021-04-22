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
        "train":
            {"guest": {"name": "nsl_kdd_train_1", "namespace": "experiment"},
             "host": [
                 {"name": "nsl_kdd_train_2", "namespace": "experiment"},
                 {"name": "nsl_kdd_train_3", "namespace": "experiment"},
                 {"name": "nsl_kdd_train_4", "namespace": "experiment"}
             ]},
        "test":
            {"guest": {"name": "nsl_kdd_test_1", "namespace": "experiment"},
             "host": [
                 {"name": "nsl_kdd_test_2", "namespace": "experiment"},
                 {"name": "nsl_kdd_test_3", "namespace": "experiment"},
                 {"name": "nsl_kdd_test_4", "namespace": "experiment"}
             ]},
        "label": "Class"
    }
    swat_ait_homo = {
        "train":
            {"guest": {"name": "swat_phys_train_p2", "namespace": "experiment_horizontal"},
             "host": [
                 {"name": "swat_phys_train_p4", "namespace": "experiment_horizontal"},
                 {"name": "swat_phys_train_p5", "namespace": "experiment_horizontal"}
             ]},
        "test":
            {"guest": {"name": "swat_phys_test_p2", "namespace": "experiment_horizontal"},
             "host": [
                 {"name": "swat_phys_test_p4", "namespace": "experiment_horizontal"},
                 {"name": "swat_phys_test_p5", "namespace": "experiment_horizontal"}
             ]},
        "label": "Normal/Attack"
    }
    swat_ait_hetero = {
        "train":
            {"guest": {"name": "swat_phys_train_p6_label", "namespace": "experiment_vertical"},
             "host": [
                 {"name": "swat_phys_train_p1", "namespace": "experiment_vertical"},
                 {"name": "swat_phys_train_p2", "namespace": "experiment_vertical"},
                 {"name": "swat_phys_train_p3", "namespace": "experiment_vertical"},
                 {"name": "swat_phys_train_p4", "namespace": "experiment_vertical"},
                 {"name": "swat_phys_train_p5", "namespace": "experiment_vertical"},
                 {"name": "swat_phys_train_p6", "namespace": "experiment_vertical"}
             ]},
        "test":
            {"guest": {"name": "swat_phys_test_p6_label", "namespace": "experiment_vertical"},
             "host": [
                 {"name": "swat_phys_test_p1", "namespace": "experiment_vertical"},
                 {"name": "swat_phys_test_p2", "namespace": "experiment_vertical"},
                 {"name": "swat_phys_test_p3", "namespace": "experiment_vertical"},
                 {"name": "swat_phys_test_p4", "namespace": "experiment_vertical"},
                 {"name": "swat_phys_test_p5", "namespace": "experiment_vertical"},
                 {"name": "swat_phys_test_p6", "namespace": "experiment_vertical"}
             ]},
        "label": "Normal/Attack"
    }