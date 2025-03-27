import numpy as np
from sklearn import cluster
from carbs.extended_targets.GGIW_Serums_Models import GGIWMixture

class DBSCANParameters:
    def __new__(cls, eps=5.0, min_samples=1, metric="euclidean", metric_params=None,\
                algorithm="auto", leaf_size=30, p=None, n_jobs=None) -> dict:
        method_params = {"method":"DBSCAN", "eps": eps, "min_samples":min_samples, "metric":metric,\
                         "metric_params":metric_params, "algorithm":algorithm, "leaf_size":leaf_size,\
                         "p":p,"n_jobs":n_jobs}
        return method_params
    
class CARBS_DBSCAN:
    def __init__(self, params:dict):
        self._dbscan = cluster.DBSCAN(eps=params["eps"], min_samples=params["min_samples"],\
                                      metric=params["metric"], metric_params=params["metric_params"],\
                                        algorithm=params["algorithm"], leaf_size=params["leaf_size"],\
                                            p=params["p"], n_jobs=params["n_jobs"])

    def cluster(self, meas_set:np.ndarray, pred_GGIW:GGIWMixture=None) -> list:
        self._dbscan.fit(meas_set.transpose())
        labels = self._dbscan.labels_
        unique_labels = set(labels)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        
        res = []
        for lab in unique_labels:
            ind = np.where(labels == lab)
            res.append(meas_set[:,np.where(labels == lab)])

        return res

    
class MeasurementClustering:
    def __init__(self, method_dict:dict):
        # Construct the clustering object with attibutes depending on the input dict
        if method_dict["method"] == "DBSCAN":
            self._clustering = CARBS_DBSCAN(method_dict)
        pass

    def cluster(self, meas_set:np.ndarray, pred_GGIW:GGIWMixture=None) -> list:
        # Cluster the measurement set to list of partition which are np.ndarrays 
        return self._clustering.cluster(meas_set, pred_GGIW)