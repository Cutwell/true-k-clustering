import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
from scipy.spatial.distance import cdist

def initial_clustering(kpred, datapoints):
    model = KMeans(n_clusters=kpred).fit(datapoints)
    pred_centroids = model.cluster_centers_
    pred_clusters = model.predict(datapoints)

    return pred_centroids, pred_clusters

def naive_initial_clustering(kpred, datapoints, seed):
    # randomly generate k centroids to start
    prev_pred_centroids = []
    if seed:
        numpy.random.seed(seed)
    pred_centroids = datapoints[numpy.random.choice(datapoints.shape[0], size=kpred), :]

    while not numpy.array_equal(prev_pred_centroids, pred_centroids):
        # compute distance matrix of points for each centroid
        distances_matrix = cdist(datapoints, pred_centroids)

        # get the index (label) of the minimum distance
        pred_clusters = distances_matrix.argmin(axis=1)

        prev_pred_centroids = numpy.copy(pred_centroids)

        for i in range(kpred):

            mask = numpy.argwhere(pred_clusters == i)
            i_labelled = datapoints[mask]

            pred_centroids[i] = numpy.array([numpy.mean(i_labelled[:,0][:,0]), numpy.mean(i_labelled[:,0][:,1])])

    return pred_centroids, pred_clusters

def average_cluster_radius(datapoints, centroids, clusters):
    k = len(set(clusters))
    radii = []
    
    # calculate distance between datapoints and all centroids
    distances = cdist(centroids, datapoints)

    for i in range(k):
        # extract distances of cluster 'i'
        mask = numpy.argwhere(clusters == i)
        i_labelled = distances[i,:][mask]
        # calculate average radius of cluster
        radii.append(numpy.average(i_labelled))

    # calculate single average radius
    r = numpy.average(radii)

    return r

def average_closest_cluster(r, centroids):
    # closest distance between clusters with a distance > 3r

    # calculate cluster distances
    distances = cdist(centroids, centroids)

    # set values below threshold to zero, to be removed by the mask without breaking the array shape
    # calculate min nonzero values of array columns
    min_distances = numpy.where(distances > (3*r), distances, numpy.inf).min(axis=1)

    d = numpy.average(min_distances) / 2

    # if no distances exceed 3r threshold, d is infinite, so return 0
    return 0 if numpy.isinf(d) else d

def probabilities(datapoints, pred_centroids, pred_clusters):
    unique, count = numpy.unique(pred_clusters, return_counts=True)
    
    return count / len(datapoints)

def cluster(datapoints, pred_centroids, p_clusters, E):
    
    # calculate distances between clusters and all datapoints
    distances = cdist(datapoints, pred_centroids)

    e_log_2 = numpy.multiply(numpy.log2(p_clusters), E)

    data_metric = numpy.subtract(distances, e_log_2)

    pred_clusters = data_metric.argmin(axis=1)

    return pred_clusters

def optimise_k(datapoints, pred_centroids, pred_clusters, E):
    timestep = 0
    prev_pred_centroids = []

    while not numpy.array_equal(prev_pred_centroids, pred_centroids):
        timestep += 1

        # calculate probability datapoint belongs to cluster
        p_clusters = probabilities(datapoints, pred_centroids, pred_clusters)

        # copy centroids into old, and reset current
        prev_pred_centroids = numpy.copy(pred_centroids)

        # cluster based on data metric
        pred_clusters = cluster(datapoints, pred_centroids, p_clusters, E)
        k = len(set(pred_clusters))     # calculate new k from labels

        pred_centroids = numpy.empty((k, 2), float)    # reset current centroids for reassignment

        # adjust centroids
        for i in range(k):
            # extract datapoints with label 'i' using mask
            mask = numpy.argwhere(pred_clusters == i)
            i_labelled = datapoints[mask]

            # calculate cluster 'i' new centroid
            pred_centroids[i] = numpy.array([numpy.mean(i_labelled[:,0][:,0]), numpy.mean(i_labelled[:,0][:,1])])

        #plt.scatter(datapoints[:,0], datapoints[:,1], c=pred_clusters)
        #plt.scatter(pred_centroids[:,0], pred_centroids[:,1], c='red')
        #plt.show()

    return timestep, pred_centroids, pred_clusters

def real_k_clustering(kpred, datapoints, true_labels, true_centers, seed=None):
    if seed:
        print(f'---\nseed: {seed}')

    # initial clustering
    initial_pred_centroids, initial_pred_clusters = initial_clustering(kpred, datapoints)
    #initial_pred_centroids, initial_pred_clusters = naive_initial_clustering(kpred, datapoints, seed)

    """
    NOTE:   initial_pred_centroids will not return in a consistent order for multiple runs.
            Further optimisation is sensitive to the order of returned intitial centroids.
            
            Therefore, centroids should be sorted by some ranking, to ensure optimal results.
    """

    # NOTE: unsorted yields k=1, k=3 in a 2:1 ratio

    # NOTE: sorting by column 0 yields k=1 only
    #initial_pred_centroids = initial_pred_centroids[initial_pred_centroids[:,0].argsort()]

    print(f'initial_pred_centroids: {initial_pred_centroids}')

    # define E
    r = average_cluster_radius(datapoints, initial_pred_centroids, initial_pred_clusters)
    half_d = average_closest_cluster(r, initial_pred_centroids)
    a = r + half_d
    E = a

    print(f'r, half_d, E: {r, half_d, E}')

    # find real k
    timestep, pred_centroids, pred_clusters = optimise_k(datapoints, initial_pred_centroids, initial_pred_clusters, E)

    print(f'timestep, pred_centroids: {timestep, pred_centroids}')

    # exit if k != true k
    #if len(pred_centroids) != len(true_centers):
    #    print('Predicted K != true K!')

    #else:
    #    # accuracy metrics
    #    centroid_accuracy = v_measure_score(true_centers.flatten(), pred_centroids.flatten())
    #    cluster_accuracy = v_measure_score(true_labels.flatten(), pred_clusters.flatten())

    #    print(f'centroid accuracy: {centroid_accuracy}')
    #    print(f'cluster accuracy: {cluster_accuracy}')

    k = pred_centroids.shape[0]

    # plot initial clustering with predicted k
    #fig, axs = plt.subplots(1, 2)
    #axs[0].set_title(f't = 0, k = {kpred}')
    #axs[0].scatter(datapoints[:,0], datapoints[:,1], c=initial_pred_clusters)
    #axs[0].scatter(initial_pred_centroids[:,0], initial_pred_centroids[:,1], c='red')
    ### plot final clustering with optimised k
    #axs[1].set_title(f't = {timestep}, k = {k}')
    #axs[1].scatter(datapoints[:,0], datapoints[:,1], c=pred_clusters)
    #axs[1].scatter(pred_centroids[:,0], pred_centroids[:,1], c='red')

    print(f'k: {k}')
    
    # return 
    return k, timestep, pred_centroids, pred_clusters, initial_pred_centroids, initial_pred_clusters

if __name__ == "__main__":

    kpred = 5
    true_k = 3

    # generate random data with n centroids
    centroids = [(-6, 2), (3, -4), (-5, 10)]
    datapoints, true_labels, true_centers = make_blobs(
        n_samples=500,
        centers=centroids,
        n_features=2,
        random_state=800,
        return_centers=True,
        center_box=(-20, 20)
    )

    k_scores = []    
    for _ in range(100):
        k, timestep, pred_centroids, pred_clusters, initial_pred_centroids, initial_pred_clusters = real_k_clustering(kpred, datapoints, true_centers, true_centers)
        k_scores.append(k)

        if k == true_k:
            # plot initial clustering with predicted k
            fig, axs = plt.subplots(1, 2)
            axs[0].set_title(f't = 0, k = {kpred}')
            axs[0].scatter(datapoints[:,0], datapoints[:,1], c=initial_pred_clusters)
            axs[0].scatter(initial_pred_centroids[:,0], initial_pred_centroids[:,1], c='red')

            # plot final clustering with optimised k
            axs[1].set_title(f't = {timestep}, k = {k}')
            axs[1].scatter(datapoints[:,0], datapoints[:,1], c=pred_clusters)
            axs[1].scatter(pred_centroids[:,0], pred_centroids[:,1], c='red')
            
            plt.show()
            break
            
    unique, count = numpy.unique(k_scores, return_counts=True)
    print(f'\n\n---\nunique, count: {str(unique), str(count)}')

    