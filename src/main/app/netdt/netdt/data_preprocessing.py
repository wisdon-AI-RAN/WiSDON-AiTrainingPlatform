import numpy as np
import json

def data_preprocessing(data_path):
    d = np.load(data_path, allow_pickle=True)

    x_sim = d["x_sim"].astype(np.float32)
    y_sim_n = d["y_sim_norm"].astype(np.float32)
    x_sim_n = d["x_sim_norm"].astype(np.float32)

    x_meas = d["x_meas"].astype(np.float32)
    y_meas_n = d["y_meas_norm"].astype(np.float32)
    x_meas_n = d["x_meas_norm"].astype(np.float32)

    x_mu = d["x_mu"].astype(np.float32)
    x_std = d["x_std"].astype(np.float32)
    y_meas_mu = d["y_meas_mu"].astype(np.float32)
    y_meas_std = d["y_meas_std"].astype(np.float32)

    feature_names = d["feature_names"].tolist()
    label_name = d["label_name"].tolist()[0]
    ru_to_pci = json.loads(str(d["ru_to_pci_json"].tolist()[0]))

    return x_sim, y_sim_n, x_sim_n, x_meas, y_meas_n, x_meas_n, x_mu, x_std, y_meas_mu, y_meas_std, feature_names, label_name, ru_to_pci