"""Example code to recreate the KESBA clustering results."""
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.clustering import KASBA
from aeon.benchmarking.results_loaders import get_estimator_results
from tsml_eval.experiments import load_and_run_clustering_experiment, \
    get_clusterer_by_name
from tsml_eval.evaluation import evaluate_clusterers_by_problem
from sklearn.metrics import rand_score, adjusted_rand_score, \
    normalized_mutual_info_score, adjusted_mutual_info_score
from aeon.benchmarking.metrics.clustering import clustering_accuracy_score

# List of all clustererers used in the experiments. Names in table 1 in comments. If
# not from aeo
clst = ["KASBA",    #KASBA
        "kmeans-ba-dtw", # DBA
        "kmeans-ba-shape_dtw",        # Shape-DBA
        "kmeans-ba-msm", # MBA
        "kmeans-euclidean", #Euclid
        "kmeans-MSM",     # MSM
        "kshape",  # k-Shape
        "ksc",     # k-SC
        "pam-msm",  # PAM-MSM
        ]

metrics = ["RI","ARI", "NMI", "Accuracy","AMI"]


def train_test(clusterer, dataset, datapath=None):
    """Run a single train/test experiment.

    Parameters
    ----------
    clusterer : str
        Name of the clusterer to use. Used to load a classifier from tsml_eval.
    dataset : str
        Name of the dataset to use. UCR datasets are listed in univariate_equal_length.
    datapath : str, optional
        Path to the data, by default None. If not present, it will be downloaded and
        stored in a directory called local_data.

    Returns
    -------
    preds : np.ndarray
        Predicted cluster labels.
    testy : np.ndarray
        True labels.
    """
    clst = get_clusterer_by_name(clusterer, random_state=0)
    trainX, trainy = load_classification(dataset, extract_path=datapath, split="train")
    testX, testy = load_classification(dataset, extract_path=datapath, split="test")
    clst.fit(trainX)
    preds = clst.predict(testX)
    return preds, testy


def combined(clusterer, dataset, datapath=None):
    """Run a single combined train/test experiment.

    Parameters
    ----------
    clusterer : str
        Name of the clusterer to use.
    dataset : str
        Name of the dataset to use.
    datapath : str, optional
        Path to the data, by default None. If not present, it will be downloaded.

    Returns
    -------
    preds : np.ndarray
        Predicted cluster labels.
    testy : np.ndarray
        True labels.
    """
    X,y = load_classification(dataset, extract_path=datapath)
    clst = get_clusterer_by_name(clusterer, random_state=0)
    pred = clst.fit_predict(X)
    return pred, y


def run_experiments(clusterers, datasets, measures, combine=False):
    """Returns a three-level dictionary
       results[clusterer][dataset][measure] = value
    """
    all_results = {}
    for c in clusterers:
        dataset_results = {}
        for d in datasets:
            if combine:
                preds, testy = combined(c, d)
            else:
                preds, testy = train_test(c, d)
            res = {}
            for m in measures:
                if m == "RI":
                    res[m] = rand_score(testy, preds)
                elif m == "ARI":
                    res[m] = adjusted_rand_score(testy, preds)
                elif m == "NMI":
                    res[m] = normalized_mutual_info_score(testy, preds)
                elif m == "Accuracy":
                    res[m] = clustering_accuracy_score(testy, preds)
                elif m == "AMI":
                    from sklearn.metrics import adjusted_mutual_info_score
                    res[m] = adjusted_mutual_info_score(testy, preds)
                else:
                    raise ValueError(f"Unknown measure {m}")
            dataset_results[d] = res
        all_results[c] = dataset_results
    return all_results


def get_published_results(clusterers,dataset, measure):
    """Get published results for a list of datasets and measures."""
    results = get_estimator_results(["KESBA","kshape"], dataset, measure)
    return results


if __name__ == "__main__":
    datasets = ["Chinatown", "GunPoint"]
    clst = ["KASBA","kshape"]
    measures = ["RI"]
    results = run_experiments(clst, datasets, measures, combine=True)
    print(results)

