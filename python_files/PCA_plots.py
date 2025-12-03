import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
import matplotlib.patches as patches
import matplotlib.transforms as transforms


def plot_pca_explained_variance(X, n_components=10, show_components=None):
    """
    Performs PCA on data X and plots explained variance ratios.

    Parameters:
    - X: Data array (samples x features)
    - n_components: Number of principal components to compute
    - show_components: number of top components to print explained variance for
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    exp_var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(exp_var_ratio)

    # Print explained variance for the first show_components if requested
    if show_components is not None:
        for i in range(min(show_components, n_components)):
            print(f"For PC {i+1}, the explained variance is: {exp_var_ratio[i]:.4f}")

    # Plot explained variance ratio per component
    plt.bar(range(1, n_components + 1), exp_var_ratio, alpha=0.5, align='center', label='Individual explained variance')
    # Plot cumulative explained variance
    plt.step(range(1, n_components + 1), cum_var_ratio, where='mid', label='Cumulative explained variance')

    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_3d_pca(X, y, label_name='labels', n_components=3, marker_size=5, title='3D PCA'):
    """
    Performs PCA on data X and creates an interactive 3D scatter plot.

    Parameters:
    - X: Data array (samples x features)
    - y: Labels array for coloring points
    - label_name: Name for the labels in the plot legend
    - n_components: Number of PCA components to reduce to (default=3)
    - marker_size: Size of scatter points
    - title: Plot title
    """
    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)

    # Create DataFrame for plotting
    df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    df[label_name] = y

    # Plot
    fig = px.scatter_3d(
        df,
        x='PC1', y='PC2', z='PC3',
        color=label_name,
        labels={'color': label_name},
        title=title
    )
    fig.update_traces(marker=dict(size=marker_size))
    fig.show()


def plot_2d_pca(
    X,
    labels,
    target_names=None,
    colors=None,
    n_components=2,
    ellipse_std=1,
    plot_centroids=True,
    save_path=None,
    save_format='svg',
    title='2D PCA Plot',
    figsize=(15, 15),
    fontsize_title=40,
    fontsize_labels=25,
    fontsize_legend=25,
    linewidth=2.5
):
    """
    Creates a 2D PCA scatter plot with optional centroids and ellipses.

    Parameters:
    - X: Data array (samples x features)
    - labels: Array or Series with labels for each sample
    - target_names: List of target names for legend (if None, inferred from labels)
    - colors: List of colors for targets; if None, uses default colormap
    - n_components: Number of PCA components (default=2)
    - ellipse_std: Number of std deviations for ellipse (default=1)
    - plot_centroids: Whether to plot centroids (default=True)
    - save_path: Path to save plot, or None to skip saving
    - save_format: 'png' or 'svg'
    - title: Plot title
    - figsize: Figure size
    - fontsize_title: Title font size
    - fontsize_labels: Axis labels font size
    - fontsize_legend: Legend font size
    - linewidth: Centroids and ellipse line width
    """
    # PCA transformation
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_

    # Convert labels to numpy array for easier indexing
    labels = np.array(labels)

    # Unique labels for targets
    if target_names is not None:
        unique_labels = target_names
    else:
        unique_labels = np.unique(labels)

    n_targets = len(unique_labels)

    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(n_targets)]
    elif isinstance(colors, list) or isinstance(colors, np.ndarray):
        # User provided colors directly
        if len(colors) != n_targets:
            raise ValueError("Number of colors must match number of target groups.")
    else:
        raise TypeError("Colors should be a list or None.")
    color_map = dict(zip(unique_labels, colors))

    plt.figure(figsize=figsize)
    plt.xticks(fontsize=fontsize_labels)
    plt.yticks(fontsize=fontsize_labels)
    plt.xlabel(f'PC1 ({explained_var[0]*100:.2f}%)', fontsize=fontsize_labels+5)
    plt.ylabel(f'PC2 ({explained_var[1]*100:.2f}%)', fontsize=fontsize_labels+5)
    plt.title(title, fontsize=fontsize_title)

    # Plot data points
    for target in unique_labels:
        indices = np.where(labels == target)[0]
        color = color_map[target]
        plt.scatter(
            principal_components[indices, 0],
            principal_components[indices, 1],
            c=[color],
            s=300,
            edgecolors='k',
            label=str(target)
        )

        # Plot centroid
        if plot_centroids:
            centroid = np.mean(principal_components[indices, :2], axis=0)
            plt.scatter(*centroid, s=2000, marker='D', c=[color], lw=2.5, edgecolors='black')
            # Draw ellipse
            plot_ellipse(principal_components[indices, :2], centroid, std_multiplier=ellipse_std,
                         edge_color=color, line_width=linewidth, alpha=1)

    plt.legend(fontsize=fontsize_legend)

    # Save plot if path provided
    if save_path:
        plt.savefig(f'{save_path}.{save_format}', format=save_format)
    plt.show()

def plot_ellipse(data, centroid, std_multiplier=1, edge_color='black', face_color='none', line_width=2.5, alpha=0.2):
    """
    Draws an ellipse at the centroid with axes scaled by std deviations.
    """
    cov = np.cov(data - centroid, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2 * std_multiplier * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    ellipse = patches.Ellipse(
        xy=centroid,
        width=width,
        height=height,
        angle=angle,
        edgecolor=edge_color,
        facecolor=face_color,
        linewidth=line_width,
        alpha=alpha
    )
    plt.gca().add_patch(ellipse)

