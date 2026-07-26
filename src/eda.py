import os

import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# GLOBAL SETTINGS
# ==========================================================

FONT = "Times New Roman"
FONT_SIZE = 15


# ==========================================================
# COMMON FORMATTING
# ==========================================================

def format_axis(ax, title="", xlabel="", ylabel=""):
    """
    Apply common formatting to every plot.
    """

    ax.set_title(
        title,
        fontsize=FONT_SIZE,
        fontname=FONT,
        fontweight="bold"
    )

    ax.set_xlabel(
        xlabel,
        fontsize=FONT_SIZE,
        fontname=FONT,
        fontweight="bold"
    )

    ax.set_ylabel(
        ylabel,
        fontsize=FONT_SIZE,
        fontname=FONT,
        fontweight="bold"
    )

    for label in ax.get_xticklabels():

        label.set_fontname(FONT)
        label.set_fontsize(FONT_SIZE)

    for label in ax.get_yticklabels():

        label.set_fontname(FONT)
        label.set_fontsize(FONT_SIZE)

    legend = ax.get_legend()

    if legend:

        for text in legend.get_texts():

            text.set_fontname(FONT)
            text.set_fontsize(FONT_SIZE)


# ==========================================================
# MISSING VALUES
# ==========================================================

def plot_missing_values(ax, df):

    missing = df.isnull().sum()

    ax.bar(
        missing.index.astype(str),
        missing.values,
        edgecolor="black"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    format_axis(
        ax,
        "Missing Values",
        "Features",
        "Count"
    )


# ==========================================================
# DATA TYPES
# ==========================================================

def plot_data_types(ax, df):

    counts = df.dtypes.astype(str).value_counts()

    ax.bar(
        counts.index,
        counts.values,
        edgecolor="black"
    )

    format_axis(
        ax,
        "Data Types",
        "Type",
        "Count"
    )


# ==========================================================
# HISTOGRAM
# ==========================================================

def plot_histogram(ax, df, column):

    values = df[column].dropna()

    mean = values.mean()

    median = values.median()

    ax.hist(
        values,
        bins=15,
        edgecolor="black"
    )

    ax.axvline(
        mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Mean"
    )

    ax.axvline(
        median,
        color="green",
        linestyle="-.",
        linewidth=2,
        label="Median"
    )

    format_axis(
        ax,
        f"{column} Histogram",
        column,
        "Frequency"
    )


# ==========================================================
# BOXPLOT
# ==========================================================

def plot_boxplot(ax, df, column):

    ax.boxplot(
        df[column].dropna(),
        tick_labels=[column]
    )

    format_axis(
        ax,
        f"{column} Boxplot",
        "Feature",
        column
    )


# ==========================================================
# SCATTER
# ==========================================================

def plot_scatter(ax, df, numeric_columns):

    if len(numeric_columns) < 2:

        ax.text(
            0.5,
            0.5,
            "Not Enough Numeric Features",
            ha="center",
            va="center"
        )

        ax.axis("off")

        return

    ax.scatter(
        df[numeric_columns[0]],
        df[numeric_columns[1]]
    )

    format_axis(
        ax,
        "Scatter Plot",
        numeric_columns[0],
        numeric_columns[1]
    )


# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

def plot_target_distribution(
    ax,
    df,
    target_column=None
):
    """
    Plot target distribution.

    If target_column is None,
    automatically use the last categorical column.
    """

    if target_column is None:

        categorical = df.select_dtypes(
            exclude=np.number
        ).columns

        if len(categorical) == 0:

            ax.text(
                0.5,
                0.5,
                "No Target Column",
                ha="center",
                va="center"
            )

            ax.axis("off")

            return

        target_column = categorical[-1]

    counts = df[target_column].value_counts()

    ax.bar(
        counts.index.astype(str),
        counts.values,
        edgecolor="black"
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )

    format_axis(
        ax,
        "Target Distribution",
        target_column,
        "Count"
    )


# ==========================================================
# HEATMAP
# ==========================================================

def plot_heatmap(ax, df):

    corr = df.corr(numeric_only=True)

    image = ax.imshow(
        corr,
        cmap="viridis"
    )

    ax.set_xticks(
        range(len(corr.columns))
    )

    ax.set_xticklabels(
        corr.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        range(len(corr.columns))
    )

    ax.set_yticklabels(
        corr.columns
    )

    for i in range(len(corr.columns)):

        for j in range(len(corr.columns)):

            ax.text(
                j,
                i,
                f"{corr.iloc[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white"
            )

    format_axis(
        ax,
        "Correlation Heatmap"
    )

    plt.colorbar(
        image,
        ax=ax
    )

# ==========================================================
# DATASET SUMMARY
# ==========================================================

def plot_summary(ax, df):

    ax.axis("off")

    numeric = len(df.select_dtypes(include=np.number).columns)
    categorical = len(df.select_dtypes(exclude=np.number).columns)

    summary = (
        f"Rows              : {df.shape[0]}\n"
        f"Columns           : {df.shape[1]}\n"
        f"Numeric Columns   : {numeric}\n"
        f"Categorical Cols  : {categorical}\n"
        f"Missing Values    : {df.isnull().sum().sum()}\n"
        f"Duplicate Rows    : {df.duplicated().sum()}"
    )

    ax.text(
        0,
        1,
        summary,
        fontsize=FONT_SIZE,
        fontname=FONT,
        verticalalignment="top"
    )

    ax.set_title(
        "Dataset Summary",
        fontsize=FONT_SIZE,
        fontname=FONT,
        fontweight="bold"
    )


# ==========================================================
# MAIN EDA FUNCTION
# ==========================================================

def perform_eda(
    df,
    target_column=None,
    save=True,
    filename="EDA_Summary"
):
    """
    Generate a 3x4 EDA summary figure.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    target_column : str, optional
        Target column name.
        If None, last categorical column is used.

    save : bool
        Save figure as EPS.

    filename : str
        Output filename.
    """

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(22, 15)
    )

    axes = axes.flatten()

    numeric_columns = list(
        df.select_dtypes(include=np.number).columns
    )

    # --------------------------------------------------
    # Plot 1
    # --------------------------------------------------

    plot_missing_values(
        axes[0],
        df
    )

    # --------------------------------------------------
    # Plot 2
    # --------------------------------------------------

    plot_data_types(
        axes[1],
        df
    )

    # --------------------------------------------------
    # Plot 3-6
    # Histograms
    # --------------------------------------------------

    histogram_axes = [2, 3, 4, 5]

    for i in range(4):

        if i < len(numeric_columns):

            plot_histogram(
                axes[histogram_axes[i]],
                df,
                numeric_columns[i]
            )

        else:

            axes[histogram_axes[i]].axis("off")

            axes[histogram_axes[i]].text(
                0.5,
                0.5,
                "Not Available",
                ha="center",
                va="center",
                fontsize=FONT_SIZE,
                fontname=FONT
            )

    # --------------------------------------------------
    # Plot 7-8
    # Boxplots
    # --------------------------------------------------

    box_axes = [6, 7]

    for i in range(2):

        if i < len(numeric_columns):

            plot_boxplot(
                axes[box_axes[i]],
                df,
                numeric_columns[i]
            )

        else:

            axes[box_axes[i]].axis("off")

            axes[box_axes[i]].text(
                0.5,
                0.5,
                "Not Available",
                ha="center",
                va="center",
                fontsize=FONT_SIZE,
                fontname=FONT
            )

    # --------------------------------------------------
    # Plot 9
    # --------------------------------------------------

    plot_scatter(
        axes[8],
        df,
        numeric_columns
    )

    # --------------------------------------------------
    # Plot 10
    # --------------------------------------------------

    plot_target_distribution(
        axes[9],
        df,
        target_column
    )

    # --------------------------------------------------
    # Plot 11
    # --------------------------------------------------

    plot_heatmap(
        axes[10],
        df
    )

    # --------------------------------------------------
    # Plot 12
    # --------------------------------------------------

    plot_summary(
        axes[11],
        df
    )

    plt.tight_layout()

    # --------------------------------------------------
    # Save Figure
    # --------------------------------------------------

    if save:

        current_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        figure_dir = os.path.join(
            current_dir,
            "..",
            "figures"
        )

        os.makedirs(
            figure_dir,
            exist_ok=True
        )

        plt.savefig(
            os.path.join(
                figure_dir,
                f"{filename}.eps"
            ),
            format="eps",
            dpi=600,
            bbox_inches="tight"
        )

    plt.show()

    # --------------------------------------------------
    # Return Summary
    # --------------------------------------------------

    summary = {

        "shape": df.shape,

        "columns": list(df.columns),

        "missing_values": df.isnull().sum(),

        "duplicates": df.duplicated().sum(),

        "numeric_columns": numeric_columns,

        "categorical_columns":
            list(
                df.select_dtypes(
                    exclude=np.number
                ).columns
            ),

        "correlation_matrix":
            df.corr(numeric_only=True)

    }

    return summary
