import numpy as np
from sklearn import cluster
from carbs.extended_targets.GGIW_Serums_Models import GGIWMixture
from scipy.stats.distributions import chi2

class DBSCANParameters:
    def __new__(cls, eps=5.0, min_samples=1, metric="euclidean", metric_params=None,\
                algorithm="auto", leaf_size=30, ignore_noise=True, p=None, n_jobs=None) -> dict:
        method_params = {"method":"DBSCAN", "eps": eps, "min_samples":min_samples, "metric":metric,\
                         "metric_params":metric_params, "algorithm":algorithm, "leaf_size":leaf_size,\
                         "ignore_noise":ignore_noise,"p":p,"n_jobs":n_jobs}
        return method_params
    
class CARBS_DBSCAN:
    def __init__(self, params:dict):
        self._dbscan = cluster.DBSCAN(eps=params["eps"], min_samples=params["min_samples"],\
                                      metric=params["metric"], metric_params=params["metric_params"],\
                                        algorithm=params["algorithm"], leaf_size=params["leaf_size"],\
                                            p=params["p"], n_jobs=params["n_jobs"])
        self._ignore_noise = params["ignore_noise"]

    def cluster(self, meas_set:list) -> list:
        res = []
        num_meas = len(meas_set)
        if num_meas == 0:
            return res
        
        meas_d = meas_set[0].shape[0]
        meas_set_array = np.array(meas_set).reshape(num_meas, meas_d)
        self._dbscan.fit(meas_set_array)
        labels = self._dbscan.labels_
        unique_labels = set(labels)
        
        for lab in unique_labels:
            if lab == -1:
                # "-1" label indicates points not part of a cluster and need special handling
                if not self._ignore_noise:
                    # If we chose not to ignore noisy point, each are assigned to their own cluster
                    meas_in_cluster = meas_set_array[np.where(labels == lab),:]
                    for ii in range(meas_in_cluster.shape[1]):
                        res.append([meas_in_cluster[:,ii,:].reshape(meas_d,1)])

            else:
                cur_cluster = []
                meas_in_cluster = meas_set_array[np.where(labels == lab),:]
                for ii in range(meas_in_cluster.shape[1]):
                    cur_cluster.append(meas_in_cluster[:,ii,:].reshape(meas_d,1))
                res.append(cur_cluster)
        return res

class DistPartitionParameters:
    def __new__(cls, max_dist_prob=0.8, min_dist_prob=0.4, meas_noise=2.0,meas_lambda=5, meas_dim=2, enable_sub=True) -> dict:
        method_params = {"method":"DISTWITHSUB", "max_dist_prob": max_dist_prob, "min_dist_prob":min_dist_prob,\
                          "meas_noise":meas_noise, "meas_lambda":meas_lambda, "meas_dim":meas_dim}
        return method_params

class DistPartition:
    def __init__(self, params:dict):
        # Use squared distance to avoid computationally expensive sqrt function
        self._max_dist_sq = params["meas_noise"] * chi2.ppf(params["max_dist_prob"],df=params["meas_dim"])
        self._max_dist_sq = self._max_dist_sq * self._max_dist_sq
        self._min_dist_sq = params["meas_noise"] * chi2.ppf(params["max_dist_prob"],df=params["meas_dim"])
        self._min_dist_sq = self._min_dist_sq * self._min_dist_sq
        self._lambda = params["meas_lambda"]

    def cluster(self, meas_set:list) -> list:
        res = []
        num_meas = len(meas_set)
        if num_meas == 0:
            return res
        
        meas_d = meas_set[0].shape[0]
        meas_set_array = np.array(meas_set).reshape(num_meas, meas_d)
        
        # Calc squared distance matrix
        x_mat = np.tile(meas_set_array[0,:],(num_meas,1))
        y_mat = np.tile(meas_set_array[1,:],(num_meas,1))
        dist_mat_sq = np.square(x_mat - x_mat.transpose()) + np.square(y_mat - y_mat.transpose())
        print("Distance partition not implemented")
        

        return res

class PoissonSubPartitionParameters:
    def __new__ (cls,predict_lambda=10):
        method_params = {"method":"PoisSub","lambda":predict_lambda}
        return method_params
    
class PoissonSubPartition:
    # Sub partition method by Karl Granstrom [doi: 10.1109/TAES.2012.6324703]
    def __init__(self, params:dict):
        self._predict_lambda = params["lambda"]

    def sub_partition(self,partition:list) -> list:
        print("Poisson sub-partition not implemented")
        return partition

class MeasurementClustering:
    def __init__(self, clust_method_dict:dict, sub_part_method_dict:dict=None):
        self._method = clust_method_dict["method"]
        # Construct the clustering object with attibutes depending on the input dict
        if self._method == "DBSCAN":
            self._clustering = CARBS_DBSCAN(clust_method_dict)
        #elif self._method == "DISTANCE":
        #    self._clustering = DistPartition(clust_method_dict)
        else:
            raise Exception("Clustering method not implemented")

        # Construct subpartition object if configured. 
        if sub_part_method_dict is not None:
            self._enable_sub = True
            self._sub_part_method = sub_part_method_dict["method"]
            if self._sub_part_method == "PoisSub":
                self._sub_partition = PoissonSubPartition(sub_part_method_dict)
        else:
            self._enable_sub = False
        pass

    def method(self):
        return self._method
    
    def subpartition_state(self):
        return self._enable_sub

    def cluster(self, meas_set:list, pred_GGIW:GGIWMixture=None) -> list:
        # Cluster the measurement set to list of partition with the following structure
        # [all_cluster] -> list
        #     |_  [cluster_1] -> list
        #     |        |_ [meas_1] -> dx1 np.array
        #     |        |_           ...
        #     |        |_ [meas_n1] -> dx1 np.array
        #     |
        #     |_  [cluster_2] -> list
        #     |        |_ [meas_1] -> dx1 np.array
        #     |        |_           ...
        #     |        |_ [meas_n2] -> dx1 np.array
        #     |_                ...
        res = self._clustering.cluster(meas_set)

        if self._enable_sub:
            if pred_GGIW is None:
                print ("Prior GGIW object is empty")
            res = self._sub_partition.sub_partition(res)
        
        return res