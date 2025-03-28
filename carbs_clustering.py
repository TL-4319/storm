import numpy as np
from sklearn import cluster
from carbs.extended_targets.GGIW_Serums_Models import GGIWMixture
from scipy.stats.distributions import chi2

class DBSCANParameters:
    def __new__(cls, eps=5.0, min_samples=1, metric="euclidean", metric_params=None,\
                algorithm="auto", leaf_size=30, p=None, n_jobs=None, enable_sub=True) -> dict:
        method_params = {"method":"DBSCAN", "eps": eps, "min_samples":min_samples, "metric":metric,\
                         "metric_params":metric_params, "algorithm":algorithm, "leaf_size":leaf_size,\
                         "p":p,"n_jobs":n_jobs, "enable_sub":enable_sub}
        return method_params
    
class CARBS_DBSCAN:
    def __init__(self, params:dict):
        self._dbscan = cluster.DBSCAN(eps=params["eps"], min_samples=params["min_samples"],\
                                      metric=params["metric"], metric_params=params["metric_params"],\
                                        algorithm=params["algorithm"], leaf_size=params["leaf_size"],\
                                            p=params["p"], n_jobs=params["n_jobs"])
        self._enable_sub = params["enable_sub"]

    def _subpartition(self, partition:list) -> list:
        pass

    def cluster(self, meas_set:np.ndarray, pred_GGIW:GGIWMixture=None) -> list:
        self._dbscan.fit(meas_set.transpose())
        labels = self._dbscan.labels_
        unique_labels = set(labels)
        
        res = []
        for lab in unique_labels:
            ind = np.where(labels == lab)
            res.append(meas_set[:,np.where(labels == lab)])

        if self._enable_sub and pred_GGIW is not None:
            res = self._enable_sub(res)
        elif self._enable_sub and pred_GGIW is None:
            print("Sub-partition requires a GGIW prior. Skipping")

        return res

class DistPartitionwithSubParameters:
    def __new__(cls, max_dist_prob=0.8, min_dist_prob=0.4, meas_noise=2.0,meas_lambda=5, meas_dim=2, enable_sub=True) -> dict:
        method_params = {"method":"DISTWITHSUB", "max_dist_prob": max_dist_prob, "min_dist_prob":min_dist_prob,\
                          "meas_noise":meas_noise, "meas_lambda":meas_lambda, "meas_dim":meas_dim, "enable_sub":enable_sub}
        return method_params

class DistPartitionwithSub:
    def __init__(self, params:dict):
        # Use squared distance to avoid computationally expensive sqrt function
        self._max_dist_sq = params["meas_noise"] * chi2.ppf(params["max_dist_prob"],df=params["meas_dim"])
        self._max_dist_sq = self._max_dist_sq * self._max_dist_sq
        self._min_dist_sq = params["meas_noise"] * chi2.ppf(params["max_dist_prob"],df=params["meas_dim"])
        self._min_dist_sq = self._min_dist_sq * self._min_dist_sq
        self._lambda = params["meas_lambda"]
        self._enable_sub = params["enable_sub"]

    def _partition(self, meas_set:np.ndarray) -> list:
        res = []
        num_meas = meas_set.shape[1]
        # Calc squared distance matrix
        x_mat = np.tile(meas_set[0,:],(num_meas,1))
        y_mat = np.tile(meas_set[1,:],(num_meas,1))
        dist_mat_sq = np.square(x_mat - x_mat.transpose()) + np.square(y_mat - y_mat.transpose())
            


        print(dist_mat_sq)

        return res

    def _subpartition(self, partition:list) -> list:
        pass

    def cluster(self, meas_set:np.ndarray, pred_GGIW:GGIWMixture=None) -> list:
        res = []
        res = self._partition(meas_set)

        if self._enable_sub and pred_GGIW is not None:
            res = self._subpartition(res)
        elif self._enable_sub and pred_GGIW is None:
            print("Sub-partition requires a GGIW prior. Skipping")

        return res

class SpectralClusteringParameters:
    def __new__(cls, n_cluster=2, eigen_solver=None, n_components=None, random_state=None, n_init=10,\
                gamma=1.0, affinity="rbf", n_neighbors=10, eigen_tol="auto", assign_labels="kmeans", degree=3, coef0=1, kernel_params=None, \
                    n_jobs=None, verbose=False, enable_sub=True) -> dict:
        method_params = {"method":"SPECTRALCLUST", "n_cluster": n_cluster, "eigen_solver":eigen_solver, "n_components":n_components,\
                         "random_state":random_state, "n_init":n_init, "gamma":gamma, "affinity":affinity, "n_neighbors":n_neighbors,"eigen_tol":eigen_tol,\
                         "assign_labels":assign_labels,"degree":degree, "coef0":coef0, "kernel_params":kernel_params,"n_jobs":n_jobs, "verbose":verbose, "enable_sub":enable_sub}
        return method_params
    
class CARBS_SpectralClustering:
    def __init__(self, params:dict):
        self._spectral_cluster = cluster.SpectralClustering(n_clusters=params["n_cluster"], eigen_solver=params["eigen_solver"], n_components=params["n_components"],\
                                                            random_state=params["random_state"], n_init=params["n_init"], gamma=params["gamma"], affinity=params["affinity"],\
                                                                n_neighbors=params["n_neighbors"], eigen_tol=params["eigen_tol"], assign_labels=params["assign_labels"],\
                                                                    degree=params["degree"], coef0=params["coef0"], kernel_params=params["kernel_params"], n_jobs=params["n_jobs"],verbose=params["verbose"])
        self._enable_sub = params["enable_sub"]

    def _subpartition(self, partition:list) -> list:
        pass

    def cluster(self, meas_set:np.ndarray, pred_GGIW:GGIWMixture=None) -> list:
        self._spectral_cluster.fit(meas_set.transpose())
        labels = self._spectral_cluster.labels_
        unique_labels = set(labels)
        
        res = []
        for lab in unique_labels:
            ind = np.where(labels == lab)
            res.append(meas_set[:,np.where(labels == lab)])

        if self._enable_sub and pred_GGIW is not None:
            res = self._enable_sub(res)
        elif self._enable_sub and pred_GGIW is None:
            print("Sub-partition requires a GGIW prior. Skipping")

        return res


class MeasurementClustering:
    def __init__(self, method_dict:dict):
        self._method = method_dict["method"]
        # Construct the clustering object with attibutes depending on the input dict
        if self._method == "DBSCAN":
            self._clustering = CARBS_DBSCAN(method_dict)
        elif self._method == "DISTWITHSUB":
            self._clustering = DistPartitionwithSub(method_dict)
        elif self._method == "SPECTRALCLUST":
            self._clustering = CARBS_SpectralClustering(method_dict)
        pass

    def method(self):
        return self._method

    def cluster(self, meas_set:np.ndarray, pred_GGIW:GGIWMixture=None) -> list:
        # Cluster the measurement set to list of partition which are np.ndarrays 
        return self._clustering.cluster(meas_set, pred_GGIW)