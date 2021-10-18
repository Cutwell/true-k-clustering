import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def naive_find_k(datapoints, kmin, kmax):

    fig, axs = plt.subplots(1, kmax-kmin+1)

    for k in range(kmin, kmax+1):

        model = KMeans(n_clusters = k).fit(datapoints)
        pred_centroids = model.cluster_centers_
        pred_clusters = model.predict(datapoints)

        axs[k-kmin].set_title(f't={model.n_iter_}, k={k}')
        axs[k-kmin].scatter(datapoints[:,0], datapoints[:,1], c=pred_clusters)
        axs[k-kmin].scatter(pred_centroids[:,0], pred_centroids[:,1], c='red')
    
    plt.show()
    
if __name__ == "__main__":
    kmin, kmax = 2, 4

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

    naive_find_k(datapoints, kmin, kmax)