"""
Helper functions for geospatial ML training notebook.
Contains plotting utilities and common operations to keep the notebook clean.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Color palettes and constants
# =============================================================================

CONTINUOUS_CMAP = 'viridis'
DIVERGING_CMAP = 'RdYlBu_r'
CATEGORICAL_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                      '#ffff33', '#a65628', '#f781bf', '#999999']
ANOMALY_COLORS = {'normal': '#34495E', 'anomaly': '#D35400'}
ALTERATION_COLORS = {
    'Background': '#f0f0f0',
    'Advanced Argillic': '#d73027',
    'Phyllic': '#fc8d59',
    'Gossan': '#fee090',
    'Argillic': '#e0f3f8',
    'Propylitic': '#91bfdb',
    'Laterite': '#4575b4'
}


# =============================================================================
# Data visualization functions
# =============================================================================

def plot_raster(data, title='Raster Data', cmap=CONTINUOUS_CMAP,
                vmin=None, vmax=None, ax=None, colorbar=True,
                extent=None, robust_stretch=True):
    """
    Plot a 2D raster array with optional colorbar.

    Parameters
    ----------
    data : np.ndarray
        2D array to plot
    title : str
        Plot title
    cmap : str
        Colormap name
    vmin, vmax : float
        Value range for colormap
    ax : matplotlib.axes.Axes
        Axes to plot on (creates new figure if None)
    colorbar : bool
        Whether to add colorbar
    extent : tuple
        (xmin, xmax, ymin, ymax) for georeferenced display
    robust_stretch : bool
        Use 2nd-98th percentile stretch

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Apply robust stretch if requested
    if robust_stretch and vmin is None and vmax is None:
        valid_data = data[~np.isnan(data)] if np.any(np.isnan(data)) else data.flatten()
        if len(valid_data) > 0:
            vmin, vmax = np.percentile(valid_data, [2, 98])

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, origin='upper')
    ax.set_title(title)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        plt.colorbar(im, cax=cax)

    return ax


def plot_vector(gdf, column=None, title='Vector Data', cmap=CONTINUOUS_CMAP,
                categorical=False, ax=None, legend=True, edgecolor='black',
                linewidth=0.5, alpha=0.7, markersize=30):
    """
    Plot a GeoDataFrame with optional column coloring.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Vector data to plot
    column : str
        Column to use for coloring
    title : str
        Plot title
    cmap : str
        Colormap for continuous data
    categorical : bool
        Whether column is categorical
    ax : matplotlib.axes.Axes
        Axes to plot on
    legend : bool
        Whether to show legend
    edgecolor : str
        Edge color for polygons
    linewidth : float
        Line width
    alpha : float
        Transparency
    markersize : int
        Size for point geometries

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    geom_type = gdf.geometry.iloc[0].geom_type

    if column is not None:
        if categorical:
            gdf.plot(column=column, ax=ax, legend=legend,
                     cmap=mcolors.ListedColormap(CATEGORICAL_COLORS[:gdf[column].nunique()]),
                     edgecolor=edgecolor, linewidth=linewidth, alpha=alpha,
                     markersize=markersize if 'Point' in geom_type else None)
        else:
            gdf.plot(column=column, ax=ax, legend=legend, cmap=cmap,
                     edgecolor=edgecolor, linewidth=linewidth, alpha=alpha,
                     markersize=markersize if 'Point' in geom_type else None)
    else:
        gdf.plot(ax=ax, edgecolor=edgecolor, linewidth=linewidth,
                 alpha=alpha, color='steelblue',
                 markersize=markersize if 'Point' in geom_type else None)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    return ax


def plot_geometry_types(gdf_point, gdf_line, gdf_poly, figsize=(15, 5)):
    """
    Plot point, line, and polygon geometries side by side.

    Parameters
    ----------
    gdf_point : geopandas.GeoDataFrame
        Point geometries
    gdf_line : geopandas.GeoDataFrame
        Line geometries
    gdf_poly : geopandas.GeoDataFrame
        Polygon geometries
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_vector(gdf_point, ax=axes[0], title='Point Geometries',
                markersize=50, legend=False)
    plot_vector(gdf_line, ax=axes[1], title='Line Geometries',
                linewidth=2, legend=False)
    plot_vector(gdf_poly, ax=axes[2], title='Polygon Geometries',
                alpha=0.6, legend=False)

    plt.tight_layout()
    return fig, axes


def plot_raster_vs_vector(raster_data, gdf, extent=None, figsize=(14, 6)):
    """
    Plot raster and vector data side by side for comparison.

    Parameters
    ----------
    raster_data : np.ndarray
        2D raster array
    gdf : geopandas.GeoDataFrame
        Vector data
    extent : tuple
        Raster extent (xmin, xmax, ymin, ymax)
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_raster(raster_data, ax=axes[0], title='Raster (Gridded)', extent=extent)
    plot_vector(gdf, ax=axes[1], title='Vector (Discrete Features)')

    plt.tight_layout()
    return fig, axes


def plot_continuous_vs_categorical(continuous_data, categorical_data,
                                   extent=None, figsize=(14, 6)):
    """
    Plot continuous and categorical rasters side by side.

    Parameters
    ----------
    continuous_data : np.ndarray
        Continuous value raster
    categorical_data : np.ndarray
        Categorical/classified raster
    extent : tuple
        Raster extent
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_raster(continuous_data, ax=axes[0], title='Continuous Data',
                cmap=CONTINUOUS_CMAP, extent=extent)

    # For categorical, use discrete colormap
    n_classes = int(np.nanmax(categorical_data)) + 1
    cmap_cat = mcolors.ListedColormap(CATEGORICAL_COLORS[:n_classes])

    im = axes[1].imshow(categorical_data, cmap=cmap_cat,
                        extent=extent, origin='upper')
    axes[1].set_title('Categorical Data')

    # Add discrete colorbar
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=range(n_classes))
    cbar.set_label('Class')

    plt.tight_layout()
    return fig, axes


# =============================================================================
# EDA and transformation plotting
# =============================================================================

def plot_distribution(data, title='Distribution', ax=None, bins=50,
                      show_stats=True, color='steelblue'):
    """
    Plot histogram with optional statistics overlay.

    Parameters
    ----------
    data : array-like
        Data to plot
    title : str
        Plot title
    ax : matplotlib.axes.Axes
        Axes to plot on
    bins : int
        Number of histogram bins
    show_stats : bool
        Whether to show mean/median/std
    color : str
        Histogram color

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    data_clean = np.array(data).flatten()
    data_clean = data_clean[~np.isnan(data_clean)]

    ax.hist(data_clean, bins=bins, color=color, alpha=0.7, edgecolor='white')
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

    if show_stats:
        mean_val = np.mean(data_clean)
        median_val = np.median(data_clean)
        std_val = np.std(data_clean)

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')

        stats_text = f'Std: {std_val:.2f}\nMin: {np.min(data_clean):.2f}\nMax: {np.max(data_clean):.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(loc='upper left')

    return ax


def plot_transformation_comparison(original, transformed, transform_name='Transformed',
                                   figsize=(14, 5)):
    """
    Plot original vs transformed data distributions.

    Parameters
    ----------
    original : array-like
        Original data
    transformed : array-like
        Transformed data
    transform_name : str
        Name of transformation applied
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_distribution(original, title='Original Distribution', ax=axes[0])
    plot_distribution(transformed, title=f'{transform_name} Distribution', ax=axes[1])

    plt.tight_layout()
    return fig, axes


def plot_correlation_matrix(df, title='Correlation Matrix', figsize=(10, 8),
                            annot=True, cmap='coolwarm'):
    """
    Plot correlation matrix heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric columns
    title : str
        Plot title
    figsize : tuple
        Figure size
    annot : bool
        Whether to annotate cells
    cmap : str
        Colormap

    Returns
    -------
    fig, ax : tuple
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=annot, fmt='.2f', square=True,
                linewidths=0.5, ax=ax)
    ax.set_title(title)

    return fig, ax


# =============================================================================
# Missing data visualization
# =============================================================================

def plot_missing_data_pattern(df, figsize=(12, 6)):
    """
    Visualize missing data patterns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Missing percentage bar chart
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)

    if len(missing_pct) > 0:
        missing_pct.plot(kind='barh', ax=axes[0], color='coral')
        axes[0].set_xlabel('Missing %')
        axes[0].set_title('Missing Data by Column')
        axes[0].axvline(x=50, color='red', linestyle='--', alpha=0.5)
    else:
        axes[0].text(0.5, 0.5, 'No missing data', ha='center', va='center',
                     transform=axes[0].transAxes, fontsize=14)
        axes[0].set_title('Missing Data by Column')

    # Missing data heatmap (sample if large)
    sample_df = df.iloc[:min(100, len(df)), :min(20, len(df.columns))]

    import seaborn as sns
    sns.heatmap(sample_df.isnull(), cbar=False, yticklabels=False,
                cmap='YlOrRd', ax=axes[1])
    axes[1].set_title('Missing Data Pattern (sample)')
    axes[1].set_xlabel('Columns')

    plt.tight_layout()
    return fig, axes


def plot_imputation_comparison(original, imputed, column_name='Value', figsize=(14, 5)):
    """
    Compare distributions before and after imputation.

    Parameters
    ----------
    original : array-like
        Original data with missing values
    imputed : array-like
        Data after imputation
    column_name : str
        Name of the variable
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Original with missing
    orig_clean = np.array(original)[~np.isnan(original)]
    axes[0].hist(orig_clean, bins=30, alpha=0.7, color='steelblue',
                 label='Original (observed)')
    axes[0].set_title(f'{column_name}: Before Imputation')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    n_missing = np.sum(np.isnan(original))
    axes[0].text(0.95, 0.95, f'N missing: {n_missing}',
                 transform=axes[0].transAxes, ha='right', va='top',
                 bbox=dict(facecolor='white', alpha=0.8))

    # After imputation
    axes[1].hist(imputed, bins=30, alpha=0.7, color='green', label='After imputation')
    axes[1].set_title(f'{column_name}: After Imputation')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    return fig, axes


# =============================================================================
# Spatial analysis visualization
# =============================================================================

def plot_semivariogram(lags, semivariance, model_fit=None, title='Semivariogram',
                       ax=None):
    """
    Plot empirical semivariogram with optional model fit.

    Parameters
    ----------
    lags : array-like
        Lag distances
    semivariance : array-like
        Semivariance values
    model_fit : array-like
        Fitted model values (optional)
    title : str
        Plot title
    ax : matplotlib.axes.Axes
        Axes to plot on

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(lags, semivariance, color='steelblue', s=50, label='Empirical')

    if model_fit is not None:
        ax.plot(lags, model_fit, color='red', linewidth=2, label='Model fit')
        ax.legend()

    ax.set_xlabel('Lag Distance')
    ax.set_ylabel('Semivariance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_spatial_autocorrelation(gdf, values, title='Spatial Autocorrelation',
                                 figsize=(14, 5)):
    """
    Visualize spatial autocorrelation with map and Moran scatterplot.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Spatial data
    values : array-like
        Values to analyze
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Spatial map of values
    gdf_plot = gdf.copy()
    gdf_plot['value'] = values
    gdf_plot.plot(column='value', ax=axes[0], legend=True, cmap=CONTINUOUS_CMAP,
                  markersize=30)
    axes[0].set_title(f'{title}: Spatial Distribution')

    # Moran scatterplot (standardized value vs spatial lag)
    from scipy.spatial import KDTree

    coords = np.array([[g.x, g.y] for g in gdf.geometry])
    tree = KDTree(coords)

    # Compute spatial lag (mean of k nearest neighbors)
    k = min(8, len(coords) - 1)
    distances, indices = tree.query(coords, k=k+1)

    values_std = (values - np.mean(values)) / np.std(values)
    spatial_lag = np.array([np.mean(values_std[idx[1:]]) for idx in indices])

    axes[1].scatter(values_std, spatial_lag, alpha=0.5, color='steelblue')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(0, color='gray', linestyle='--', alpha=0.5)

    # Add regression line
    z = np.polyfit(values_std, spatial_lag, 1)
    p = np.poly1d(z)
    x_line = np.linspace(values_std.min(), values_std.max(), 100)
    axes[1].plot(x_line, p(x_line), 'r-', linewidth=2,
                 label=f'Slope (Moran\'s I proxy): {z[0]:.3f}')

    axes[1].set_xlabel('Standardized Value')
    axes[1].set_ylabel('Spatial Lag')
    axes[1].set_title('Moran Scatterplot')
    axes[1].legend()

    plt.tight_layout()
    return fig, axes


# =============================================================================
# ML results visualization
# =============================================================================

def plot_interpolation_results(original_points, original_values,
                               interpolated_grid, extent, method_name='Interpolation',
                               figsize=(14, 5)):
    """
    Plot interpolation input and output side by side.

    Parameters
    ----------
    original_points : array-like
        (N, 2) array of point coordinates
    original_values : array-like
        Values at original points
    interpolated_grid : np.ndarray
        2D interpolated grid
    extent : tuple
        (xmin, xmax, ymin, ymax) for grid
    method_name : str
        Name of interpolation method
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Original points
    sc = axes[0].scatter(original_points[:, 0], original_points[:, 1],
                         c=original_values, cmap=CONTINUOUS_CMAP,
                         s=50, edgecolor='white')
    axes[0].set_title('Sample Points')
    plt.colorbar(sc, ax=axes[0], shrink=0.8)

    # Interpolated surface
    plot_raster(interpolated_grid, ax=axes[1], title=f'{method_name} Result',
                extent=extent, robust_stretch=False)

    # Overlay points on interpolated surface
    axes[1].scatter(original_points[:, 0], original_points[:, 1],
                    c='red', s=10, alpha=0.5, marker='.')

    plt.tight_layout()
    return fig, axes


def plot_pca_results(pca, feature_names, figsize=(14, 5)):
    """
    Plot PCA explained variance and loadings.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA object
    feature_names : list
        Names of original features
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Explained variance
    n_components = len(pca.explained_variance_ratio_)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    axes[0].bar(range(1, n_components+1), pca.explained_variance_ratio_,
                alpha=0.7, label='Individual')
    axes[0].plot(range(1, n_components+1), cum_var, 'ro-', label='Cumulative')
    axes[0].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('PCA Explained Variance')
    axes[0].legend()
    axes[0].set_xticks(range(1, n_components+1))

    # Loadings heatmap
    n_show = min(3, n_components)
    loadings = pd.DataFrame(
        pca.components_[:n_show].T,
        columns=[f'PC{i+1}' for i in range(n_show)],
        index=feature_names
    )

    import seaborn as sns
    sns.heatmap(loadings, cmap='RdBu_r', center=0, annot=True,
                fmt='.2f', ax=axes[1])
    axes[1].set_title('PCA Loadings')

    plt.tight_layout()
    return fig, axes


def plot_clustering_results(data, labels, centers=None, title='Clustering Results',
                            ax=None, feature_x=0, feature_y=1):
    """
    Plot 2D clustering results.

    Parameters
    ----------
    data : np.ndarray
        (N, D) data array
    labels : array-like
        Cluster labels
    centers : np.ndarray
        Cluster centers (optional)
    title : str
        Plot title
    ax : matplotlib.axes.Axes
        Axes to plot on
    feature_x, feature_y : int
        Feature indices for x and y axes

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    n_clusters = len(np.unique(labels))
    colors = CATEGORICAL_COLORS[:n_clusters]

    for i, color in enumerate(colors):
        mask = labels == i
        ax.scatter(data[mask, feature_x], data[mask, feature_y],
                   c=color, label=f'Cluster {i}', alpha=0.6, s=30)

    if centers is not None:
        ax.scatter(centers[:, feature_x], centers[:, feature_y],
                   c='black', marker='X', s=200, edgecolor='white',
                   linewidth=2, label='Centers')

    ax.set_xlabel(f'Feature {feature_x}')
    ax.set_ylabel(f'Feature {feature_y}')
    ax.set_title(title)
    ax.legend()

    return ax


def plot_elbow_silhouette(k_range, inertias, silhouettes, figsize=(14, 5)):
    """
    Plot elbow curve and silhouette scores for K-means.

    Parameters
    ----------
    k_range : array-like
        Range of k values tested
    inertias : array-like
        Inertia values for each k
    silhouettes : array-like
        Silhouette scores for each k
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Elbow plot
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True, alpha=0.3)

    # Silhouette plot
    axes[1].plot(k_range, silhouettes, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].grid(True, alpha=0.3)

    # Mark best k
    best_k_idx = np.argmax(silhouettes)
    axes[1].axvline(k_range[best_k_idx], color='red', linestyle='--',
                    label=f'Best k={k_range[best_k_idx]}')
    axes[1].legend()

    plt.tight_layout()
    return fig, axes


def plot_anomaly_scores(gdf, scores, binary_labels=None, title='Anomaly Detection',
                        figsize=(14, 5)):
    """
    Plot anomaly scores as continuous and binary classification.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Spatial data with geometries
    scores : array-like
        Continuous anomaly scores
    binary_labels : array-like
        Binary labels (1=anomaly, -1 or 0=normal), optional
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    gdf_plot = gdf.copy()
    gdf_plot['score'] = scores

    # Continuous scores
    gdf_plot.plot(column='score', ax=axes[0], legend=True, cmap='YlOrRd',
                  markersize=30)
    axes[0].set_title(f'{title}: Anomaly Scores')

    # Binary classification
    if binary_labels is not None:
        gdf_plot['is_anomaly'] = binary_labels

        normal_mask = gdf_plot['is_anomaly'] <= 0
        anomaly_mask = gdf_plot['is_anomaly'] > 0

        gdf_plot[normal_mask].plot(ax=axes[1], color=ANOMALY_COLORS['normal'],
                                    markersize=30, label='Normal')
        gdf_plot[anomaly_mask].plot(ax=axes[1], color=ANOMALY_COLORS['anomaly'],
                                     markersize=50, label='Anomaly')
        axes[1].legend()
    else:
        # Use threshold on scores
        threshold = np.percentile(scores, 95)
        normal_mask = scores < threshold

        gdf_plot[normal_mask].plot(ax=axes[1], color=ANOMALY_COLORS['normal'],
                                    markersize=30, label='Normal')
        gdf_plot[~normal_mask].plot(ax=axes[1], color=ANOMALY_COLORS['anomaly'],
                                     markersize=50, label='Anomaly (top 5%)')
        axes[1].legend()

    axes[1].set_title(f'{title}: Classification')

    plt.tight_layout()
    return fig, axes


def plot_alteration_map(class_map, class_names=None, figsize=(10, 8)):
    """
    Plot alteration type classification map.

    Parameters
    ----------
    class_map : np.ndarray
        2D array with class labels (0=background, 1-6=alteration types)
    class_names : list
        Names for each class
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : tuple
    """
    if class_names is None:
        class_names = list(ALTERATION_COLORS.keys())

    fig, ax = plt.subplots(figsize=figsize)

    colors = list(ALTERATION_COLORS.values())
    cmap = mcolors.ListedColormap(colors[:len(class_names)])

    im = ax.imshow(class_map, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    ax.set_title('Alteration Type Classification')

    # Create legend
    patches = [Patch(facecolor=colors[i], label=class_names[i])
               for i in range(len(class_names))]
    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    return fig, ax


def plot_prospectivity_map(prob_map, extent=None, threshold=0.5,
                           figsize=(14, 5)):
    """
    Plot prospectivity probability map with thresholded classification.

    Parameters
    ----------
    prob_map : np.ndarray
        2D probability array
    extent : tuple
        (xmin, xmax, ymin, ymax) for display
    threshold : float
        Classification threshold
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Probability map
    im0 = axes[0].imshow(prob_map, cmap='RdYlGn', vmin=0, vmax=1,
                         extent=extent, origin='upper')
    axes[0].set_title('Prospectivity Probability')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Thresholded
    classified = (prob_map >= threshold).astype(int)
    cmap_binary = mcolors.ListedColormap(['#d73027', '#1a9850'])

    im1 = axes[1].imshow(classified, cmap=cmap_binary,
                         extent=extent, origin='upper')
    axes[1].set_title(f'Prospective Areas (threshold={threshold})')

    patches = [Patch(facecolor='#d73027', label='Low'),
               Patch(facecolor='#1a9850', label='High')]
    axes[1].legend(handles=patches, loc='upper right')

    plt.tight_layout()
    return fig, axes


def plot_roc_pr_curves(y_true, y_prob, figsize=(14, 5)):
    """
    Plot ROC and Precision-Recall curves.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : tuple
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    axes[0].plot(fpr, tpr, color='steelblue', linewidth=2,
                 label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = np.mean(y_true)

    axes[1].plot(recall, precision, color='steelblue', linewidth=2,
                 label=f'PR (AP = {ap:.3f})')
    axes[1].axhline(baseline, color='gray', linestyle='--',
                    label=f'Baseline = {baseline:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_feature_importance(importance_df, top_n=15, figsize=(10, 6)):
    """
    Plot feature importance bar chart.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : tuple
    """
    fig, ax = plt.subplots(figsize=figsize)

    df_sorted = importance_df.nlargest(top_n, 'importance')

    ax.barh(range(len(df_sorted)), df_sorted['importance'], color='steelblue')
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.invert_yaxis()

    plt.tight_layout()
    return fig, ax


# =============================================================================
# Synthetic data generation
# =============================================================================

def generate_synthetic_geochemistry(n_samples=500, n_elements=10, n_clusters=4,
                                    spatial_extent=(0, 1000, 0, 1000),
                                    cluster_std=100, random_state=42):
    """
    Generate synthetic geochemistry point data with spatial clustering.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_elements : int
        Number of geochemical elements
    n_clusters : int
        Number of distinct geochemical populations
    spatial_extent : tuple
        (xmin, xmax, ymin, ymax)
    cluster_std : float
        Spatial spread of clusters
    random_state : int
        Random seed

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Synthetic geochemistry data
    """
    import geopandas as gpd
    from shapely.geometry import Point

    np.random.seed(random_state)

    xmin, xmax, ymin, ymax = spatial_extent

    # Generate cluster centers
    cluster_centers_xy = np.column_stack([
        np.random.uniform(xmin + cluster_std, xmax - cluster_std, n_clusters),
        np.random.uniform(ymin + cluster_std, ymax - cluster_std, n_clusters)
    ])

    # Generate distinct geochemical signatures for each cluster
    element_names = [f'Element_{i+1}' for i in range(n_elements)]
    cluster_signatures = np.random.lognormal(mean=3, sigma=1,
                                              size=(n_clusters, n_elements))

    # Assign samples to clusters
    samples_per_cluster = n_samples // n_clusters

    all_coords = []
    all_values = []
    all_labels = []

    for c in range(n_clusters):
        n_c = samples_per_cluster if c < n_clusters - 1 else n_samples - len(all_coords)

        # Spatial positions (clustered)
        coords = np.column_stack([
            np.random.normal(cluster_centers_xy[c, 0], cluster_std, n_c),
            np.random.normal(cluster_centers_xy[c, 1], cluster_std, n_c)
        ])

        # Geochemical values (based on signature + noise)
        values = cluster_signatures[c] * np.random.lognormal(0, 0.3, (n_c, n_elements))

        all_coords.append(coords)
        all_values.append(values)
        all_labels.extend([c] * n_c)

    coords = np.vstack(all_coords)
    values = np.vstack(all_values)

    # Clip to extent
    coords[:, 0] = np.clip(coords[:, 0], xmin, xmax)
    coords[:, 1] = np.clip(coords[:, 1], ymin, ymax)

    # Create GeoDataFrame
    df = pd.DataFrame(values, columns=element_names)
    df['X'] = coords[:, 0]
    df['Y'] = coords[:, 1]
    df['true_cluster'] = all_labels

    geometry = [Point(x, y) for x, y in coords]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:32610')

    return gdf


def generate_synthetic_raster(shape=(200, 200), extent=(0, 1000, 0, 1000),
                              n_anomalies=5, anomaly_radius=30,
                              background_mean=10, anomaly_strength=50,
                              noise_level=2, random_state=42):
    """
    Generate synthetic continuous raster with anomalies.

    Parameters
    ----------
    shape : tuple
        (rows, cols) of output raster
    extent : tuple
        (xmin, xmax, ymin, ymax) spatial extent
    n_anomalies : int
        Number of anomalous regions
    anomaly_radius : float
        Radius of anomalies in grid cells
    background_mean : float
        Mean background value
    anomaly_strength : float
        Added value at anomaly centers
    noise_level : float
        Standard deviation of noise
    random_state : int
        Random seed

    Returns
    -------
    data : np.ndarray
        2D raster array
    extent : tuple
        Spatial extent
    """
    np.random.seed(random_state)

    rows, cols = shape

    # Background with spatial trend
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    xx, yy = np.meshgrid(x, y)

    # Gentle spatial trend
    data = background_mean + 2 * np.sin(2 * np.pi * xx) + 2 * np.cos(2 * np.pi * yy)

    # Add anomalies
    for _ in range(n_anomalies):
        cx = np.random.randint(anomaly_radius, cols - anomaly_radius)
        cy = np.random.randint(anomaly_radius, rows - anomaly_radius)

        # Gaussian anomaly
        for i in range(rows):
            for j in range(cols):
                dist = np.sqrt((i - cy)**2 + (j - cx)**2)
                if dist < anomaly_radius * 2:
                    data[i, j] += anomaly_strength * np.exp(-dist**2 / (2 * anomaly_radius**2))

    # Add noise
    data += np.random.normal(0, noise_level, shape)

    return data, extent


def generate_synthetic_vector_geometries(n_points=50, n_lines=10, n_polygons=8,
                                         extent=(0, 1000, 0, 1000), random_state=42):
    """
    Generate synthetic point, line, and polygon geometries.

    Parameters
    ----------
    n_points : int
        Number of points
    n_lines : int
        Number of line features
    n_polygons : int
        Number of polygon features
    extent : tuple
        (xmin, xmax, ymin, ymax)
    random_state : int
        Random seed

    Returns
    -------
    gdf_points, gdf_lines, gdf_polygons : tuple of GeoDataFrames
    """
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon

    np.random.seed(random_state)
    xmin, xmax, ymin, ymax = extent

    # Points
    points = [Point(np.random.uniform(xmin, xmax),
                    np.random.uniform(ymin, ymax)) for _ in range(n_points)]
    gdf_points = gpd.GeoDataFrame({'id': range(n_points),
                                    'value': np.random.uniform(0, 100, n_points)},
                                   geometry=points, crs='EPSG:32610')

    # Lines (random walks)
    lines = []
    for _ in range(n_lines):
        start = [np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)]
        coords = [start]
        for _ in range(np.random.randint(3, 8)):
            new_pt = [coords[-1][0] + np.random.normal(0, 50),
                      coords[-1][1] + np.random.normal(0, 50)]
            new_pt[0] = np.clip(new_pt[0], xmin, xmax)
            new_pt[1] = np.clip(new_pt[1], ymin, ymax)
            coords.append(new_pt)
        lines.append(LineString(coords))

    gdf_lines = gpd.GeoDataFrame({'id': range(n_lines),
                                   'length': [l.length for l in lines]},
                                  geometry=lines, crs='EPSG:32610')

    # Polygons (random convex shapes)
    polygons = []
    for _ in range(n_polygons):
        cx = np.random.uniform(xmin + 50, xmax - 50)
        cy = np.random.uniform(ymin + 50, ymax - 50)
        n_vertices = np.random.randint(4, 8)
        angles = np.sort(np.random.uniform(0, 2*np.pi, n_vertices))
        radii = np.random.uniform(20, 80, n_vertices)
        coords = [(cx + r * np.cos(a), cy + r * np.sin(a))
                  for a, r in zip(angles, radii)]
        coords.append(coords[0])  # close polygon
        polygons.append(Polygon(coords))

    gdf_polygons = gpd.GeoDataFrame({'id': range(n_polygons),
                                      'area': [p.area for p in polygons]},
                                     geometry=polygons, crs='EPSG:32610')

    return gdf_points, gdf_lines, gdf_polygons


def add_missing_data(df, missing_pct=0.1, columns=None, pattern='random',
                     random_state=42):
    """
    Add missing values to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    missing_pct : float
        Percentage of values to make missing (0-1)
    columns : list
        Columns to add missing values to (None = all numeric)
    pattern : str
        'random' or 'spatial' (clustered missing)
    random_state : int
        Random seed

    Returns
    -------
    df_missing : pd.DataFrame
        DataFrame with missing values
    """
    np.random.seed(random_state)
    df_missing = df.copy()

    if columns is None:
        columns = df_missing.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in ['X', 'Y', 'geometry']:
            continue

        n_missing = int(len(df_missing) * missing_pct)

        if pattern == 'random':
            missing_idx = np.random.choice(df_missing.index, n_missing, replace=False)
        else:  # spatial clustering of missing values
            # Cluster missing values in one corner
            if 'X' in df_missing.columns and 'Y' in df_missing.columns:
                corner_mask = (df_missing['X'] < df_missing['X'].median()) & \
                              (df_missing['Y'] < df_missing['Y'].median())
                corner_idx = df_missing[corner_mask].index
                if len(corner_idx) >= n_missing:
                    missing_idx = np.random.choice(corner_idx, n_missing, replace=False)
                else:
                    missing_idx = np.random.choice(df_missing.index, n_missing, replace=False)
            else:
                missing_idx = np.random.choice(df_missing.index, n_missing, replace=False)

        df_missing.loc[missing_idx, col] = np.nan

    return df_missing
