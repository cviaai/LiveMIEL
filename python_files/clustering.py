import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score


def hopkins_statistic(X, sample_size=None, random_state=None):
    """
    Calculate the Hopkins statistic for cluster tendency assessment.
    
    The Hopkins statistic measures the probability that a dataset is generated
    by a uniform random distribution. Values close to 0.5 suggest random data
    (close to 1.0 - uniformly distributed data), while values close to 0.0 
    indicate highly clustered data.
    
    Parameters
    ----------
    - X: array-like of shape (n_samples, n_features)
      The input data matrix
    - sample_size: int, optional
      Number of samples to use for the test. If None, uses min(0.1 * n_samples, 100)
    - random_state: int or RandomState, optional
      Random state for reproducible results
        
    Returns
    -------
    - (float)
       Hopkins statistic between 0 and 1
        
    Notes
    -----
    Hopkins statistic formula:
        H = Σw_i / (Σu_i + Σw_i)
    where:
        u_i: distance from uniform random point to nearest actual data point
        w_i: distance from actual data point to nearest other data point
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    
    n_samples, n_features = X.shape
    
    if sample_size is None:
        sample_size = min(int(0.1 * n_samples), 100)
    elif sample_size >= n_samples:
        raise ValueError("sample_size must be less than number of samples")
    
    random_state = check_random_state(random_state)
    
    # Fit nearest neighbors on the actual data
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X)
    
    # Sample actual data points (for w distances)
    actual_indices = random_state.choice(n_samples, size=sample_size, replace=False)
    
    u_distances = []  # distances from uniform points to actual data
    w_distances = []  # distances from actual points to other actual data
    
    # Get data bounds for uniform sampling
    data_min = np.min(X, axis=0)
    data_max = np.max(X, axis=0)
    data_range = data_max - data_min
    
    for i in range(sample_size):
        # Generate uniform random point in data space
        uniform_point = random_state.random(n_features) * data_range + data_min
        
        # Distance from uniform point to nearest actual data point (u_i)
        u_dist, _ = nbrs.kneighbors(uniform_point.reshape(1, -1), n_neighbors=2)
        u_distances.append(u_dist[0, 1])
        
        # Distance from actual point to nearest other actual point (w_i)
        actual_point = X[actual_indices[i]].reshape(1, -1)
        w_dist, _ = nbrs.kneighbors(actual_point, n_neighbors=2)
        w_distances.append(w_dist[0, 1]) 
    
    sum_u = np.sum(u_distances)
    sum_w = np.sum(w_distances)
    
    # Handle edge cases
    if sum_u + sum_w == 0:
        return 0.5  # Indeterminate case, return neutral value
    
    statistic = sum_w / (sum_u + sum_w)
    
    # Ensure result is valid
    return np.clip(statistic, 0.0, 1.0)

def silhouette_analysis(
    X,
    labels=None,
    clustering_method=None, #{None, 'kmeans', 'gmm'}
    n_clusters=None,
    random_state=42,
    init = 'k-means++', #{‘k-means++’, ‘random’}
    n_init = 1, #{‘auto’, int}
    show_plot=True,
    save_path=None,
    save_format='png',
    fontsize_title=18,
    fontsize_subtitle=18,
    fontsize_axis=15,
    fontsize_labels=12,
    marker_size=170
):
    """
    Perform silhouette analysis on data X with optional labels.
    Supports KMeans and GMM clustering.

    Parameters:
    - X: data (samples x features)
    - labels: true labels (optional, for reference), if None, unlabeled
    - clustering_method: 'kmeans' or 'gmm'
    - n_clusters: number of clusters
    - random_state: for reproducibility
    - n_init: 
    - show_plot: whether to display plots
    - save_path: path to save the figure; if None, does not save
    - save_format: 'png' or 'svg'
    - fontsize_title: Title font size
    - fontsize_subtitle: Subtitle font size
    - fontsize_axis: Axis labels font size
    - fontsize_labels: Tick labels font size
    - marker_size: Size of scatter points
    """
    # Use provided labels as cluster labels if no clustering method is specified
    if labels is not None and clustering_method is None:
        cluster_labels = np.array(labels)
        n_clusters = len(np.unique(cluster_labels))
        centers = compute_cluster_centers(X, cluster_labels)
        method_desc = 'Labeled data'
    elif clustering_method is not None:
        # Clustering
        if clustering_method == 'kmeans':
            if n_clusters is None:
                raise ValueError("n_clusters must be specified for KMeans.")
            clusterer = KMeans(n_clusters=n_clusters, 
                               init = init, 
                               n_init=n_init, 
                               random_state=random_state)
            cluster_labels = clusterer.fit_predict(X)
            centers = clusterer.cluster_centers_
        elif clustering_method == 'gmm':
            if n_clusters is None:
                raise ValueError("n_clusters must be specified for GMM.")
            clusterer = GaussianMixture(n_components=n_clusters, 
                                        init_params = init,
                                        covariance_type='full', 
                                        random_state=random_state)
            cluster_labels = clusterer.fit_predict(X)
            centers = clusterer.means_
        else:
            raise ValueError("clustering_method must be 'kmeans', 'gmm', or None.")
        n_clusters = len(np.unique(cluster_labels))
        method_desc = clustering_method.upper()
    else:
        # Neither labels nor clustering method specified
        raise ValueError("Provide either 'labels' for labeled data or specify 'clustering_method'.")

    # Compute silhouette scores
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # Determine x-axis range with margins based on silhouette values
    sil_min, sil_max = np.min(sample_silhouette_values), np.max(sample_silhouette_values)
    margin = 0.05 * (sil_max - sil_min) if sil_max != sil_min else 0.1
    x_min = sil_min - margin
    x_max = sil_max + margin

    # Plot configuration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Colors
    cmap = plt.get_cmap('nipy_spectral')
    cluster_colors = [cmap(i / n_clusters) for i in range(n_clusters)]
    color_map = dict(zip(range(n_clusters), cluster_colors))

    # Handle arbitrary labels for silhouette plot and data coloring
    if labels is not None and clustering_method is None:
        # Map arbitrary labels to integers
        label_mapping, sample_label_indices = create_label_mapping(labels)
        # Plot silhouette segments grouped by original labels
        plot_silhouette_segments(
            ax=ax1,
            sample_silhouette_values=sample_silhouette_values,
            labels=labels,
            label_mapping=label_mapping,
            cluster_colors=cluster_colors
        )
        # Assign colors for data points
        point_colors = [cluster_colors[idx] for idx in sample_label_indices]
    else:
        # Clustering case
        plot_silhouette_segments(
            ax=ax1,
            sample_silhouette_values=sample_silhouette_values,
            labels=cluster_labels,
            label_mapping=None,
            cluster_colors=cluster_colors,
            labels_size=fontsize_labels
        )
        # Assign colors for data points
        point_colors = [color_map[label] for label in cluster_labels]

    ax1.set_title("Silhouette plot for clusters", size=fontsize_subtitle)
    ax1.set_xlabel("Silhouette coefficient values", size=fontsize_axis)
    ax1.set_ylabel("Cluster label", size=fontsize_axis)
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])

    # Calculate the starting point (closest higher or equal multiple of 0.1)
    start_tick = np.ceil(x_min * 10) / 10
    # Calculate the ending point (closest lower or equal multiple of 0.1)
    end_tick = np.floor(x_max * 10) / 10
    # Generate ticks
    tick_positions = np.arange(start_tick, end_tick + 0.1, 0.1)

    ax1.set_xticks(tick_positions)
    ax1.tick_params(axis='x', labelsize=fontsize_labels)

    # Plot data with cluster coloring
    ax2.scatter(X[:, 0], X[:, 1], c=point_colors, s=marker_size, edgecolors='k', alpha=0.7)

    # Plot cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c='white', s=200, edgecolor='k')
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker=f"${i}$", s=50, edgecolor='k', c='black')

    ax2.set_title("Clustered data visualization", size=fontsize_subtitle)
    ax2.tick_params(axis='both', labelsize=fontsize_labels)
    ax2.set_xlabel("PC1", size=fontsize_axis)
    ax2.set_ylabel("PC2", size=fontsize_axis)

    plt.suptitle(
        f"Silhouette analysis ({method_desc}) with n_clusters={n_clusters}",
        fontsize=fontsize_title, fontweight='bold'
    )

    # Save figure if path provided
    if save_path:
        plt.savefig(f"{save_path}.{save_format}", format=save_format, bbox_inches='tight')

    if show_plot:
        plt.show()

def compute_cluster_centers(X, labels):
    """
    Compute the mean (center) of data points for each cluster label.
    """
    unique_labels = np.unique(labels)
    centers = np.zeros((len(unique_labels), X.shape[1]))
    for i, lbl in enumerate(unique_labels):
        centers[i] = X[labels == lbl].mean(axis=0)
    return centers

def create_label_mapping(labels):
    """
    Map arbitrary labels to integer indices for plotting and coloring.

    Returns:
    - label_mapping: dict of label -> index
    - sample_label_indices: list of indices corresponding to each label in labels
    """
    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    sample_label_indices = [label_mapping[label] for label in labels]
    return label_mapping, sample_label_indices

def plot_silhouette_segments(
    ax,
    sample_silhouette_values,
    labels, 
    label_mapping,
    cluster_colors,
    labels_size=12
):

    """
    Plot silhouette segments grouped by labels or cluster labels.
    """
    y_lower = 10
    if label_mapping is not None:
        # labels are arbitrary
        for label_value, label_idx in label_mapping.items():
            mask = (np.array(labels) == label_value)
            values = sample_silhouette_values[mask]
            values.sort()
            size = values.shape[0]
            y_upper = y_lower + size
            color = cluster_colors[label_idx]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ax.text(-0.07, y_lower + 0.5 * size, str(label_value), size=labels_size)
            y_lower = y_upper + 10
    else:
        # labels are cluster labels
        n_clusters = len(cluster_colors)
        for i in range(n_clusters):
            cluster_silhouette_values = sample_silhouette_values[labels == i]
            cluster_silhouette_values = np.sort(cluster_silhouette_values)
            size = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size
            color = cluster_colors[i]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ax.text(-0.05, y_lower + 0.5 * size, str(i), size=labels_size)
            y_lower = y_upper + 10
