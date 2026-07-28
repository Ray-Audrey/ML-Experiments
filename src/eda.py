import os
import numpy as np
import matplotlib.pyplot as plt


FONT = "Times New Roman"
FONT_SIZE = 15



# ==========================================================
# FORMAT AXIS
# ==========================================================

def format_axis(
    ax,
    title,
    xlabel="",
    ylabel=""
):

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





# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

def plot_target_distribution(
    df,
    target_column,
    save_dir="../figures"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    values = df[target_column].dropna()



    plt.figure(
        figsize=(7,5)
    )


    plt.hist(
        values,
        bins=30,
        edgecolor="black"
    )


    format_axis(
        plt.gca(),
        "Target Variable Distribution",
        target_column,
        "Frequency"
    )


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "target_distribution.png"
        ),
        dpi=600
    )


    plt.show()





# ==========================================================
# FEATURE VS TARGET SCATTER
# ==========================================================

def plot_feature_target(
    df,
    target_column,
    save_dir="../figures",
    max_features=5
):


    numeric_features = [

        col

        for col in df.select_dtypes(
            include=np.number
        ).columns

        if col != target_column

    ]



    for feature in numeric_features[:max_features]:


        plt.figure(
            figsize=(7,5)
        )


        plt.scatter(

            df[feature],

            df[target_column],

            s=15

        )


        format_axis(

            plt.gca(),

            f"{feature} vs {target_column}",

            feature,

            target_column

        )


        plt.grid(True)



        plt.tight_layout()



        plt.savefig(

            os.path.join(

                save_dir,

                f"{feature}_vs_target.png"

            ),

            dpi=600

        )


        plt.show()





# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

def plot_correlation_heatmap(
    df,
    save_dir="../figures"
):


    correlation = df.corr(
        numeric_only=True
    )


    plt.figure(
        figsize=(10,8)
    )


    plt.imshow(

        correlation,

        cmap="viridis"

    )


    plt.colorbar()



    plt.xticks(

        range(
            len(correlation.columns)
        ),

        correlation.columns,

        rotation=90,

        fontsize=8

    )


    plt.yticks(

        range(
            len(correlation.columns)
        ),

        correlation.columns,

        fontsize=8

    )


    plt.title(

        "Feature Correlation Heatmap",

        fontsize=FONT_SIZE,

        fontname=FONT,

        fontweight="bold"

    )


    plt.tight_layout()



    plt.savefig(

        os.path.join(

            save_dir,

            "correlation_heatmap.png"

        ),

        dpi=600

    )


    plt.show()





# ==========================================================
# CORRELATION WITH TARGET
# ==========================================================

def target_correlation(
    df,
    target_column
):


    correlation = (

        df.corr(
            numeric_only=True
        )[target_column]

        .sort_values(
            ascending=False
        )

    )


    return correlation



# ==========================================================
# COMPLETE EDA SUMMARY FIGURE
# ==========================================================

def plot_eda_summary(
    df,
    target_column,
    save_dir="../figures"
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    numeric_features = [
        col
        for col in df.select_dtypes(include=np.number).columns
        if col != target_column
    ]


    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14,18)
    )


    # Target distribution
    axes[0,0].hist(
        df[target_column].dropna(),
        bins=30,
        edgecolor="black"
    )

    format_axis(
        axes[0,0],
        "Target Distribution",
        target_column,
        "Frequency"
    )


    # Age distribution
    if "Age" in df.columns:

        axes[0,1].hist(
            df["Age"].dropna(),
            bins=30,
            edgecolor="black"
        )

        format_axis(
            axes[0,1],
            "Age Distribution",
            "Age",
            "Frequency"
        )


    # Income vs Target
    if "Income (USD)" in df.columns:

        axes[1,0].scatter(
            df["Income (USD)"],
            df[target_column],
            s=10
        )

        format_axis(
            axes[1,0],
            "Income vs Target",
            "Income",
            target_column
        )


    # Loan request vs Target
    if "Loan Amount Request (USD)" in df.columns:

        axes[1,1].scatter(
            df["Loan Amount Request (USD)"],
            df[target_column],
            s=10
        )

        format_axis(
            axes[1,1],
            "Loan Request vs Target",
            "Loan Request",
            target_column
        )


    # Correlation heatmap
    correlation = df.corr(
        numeric_only=True
    )

    axes[2,0].imshow(
        correlation,
        cmap="viridis"
    )

    axes[2,0].set_title(
        "Correlation Heatmap",
        fontsize=FONT_SIZE,
        fontweight="bold"
    )


    # Credit score vs target
    if "Credit Score" in df.columns:

        axes[2,1].scatter(
            df["Credit Score"],
            df[target_column],
            s=10
        )

        format_axis(
            axes[2,1],
            "Credit Score vs Target",
            "Credit Score",
            target_column
        )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            "exp3_eda_summary.png"
        ),
        dpi=600,
        bbox_inches="tight"
    )


    plt.show()

# ==========================================================
# COMPLETE REGRESSION EDA
# ==========================================================

def perform_regression_eda(
    df,
    target_column,
    save_dir="../figures"
):


    os.makedirs(
        save_dir,
        exist_ok=True
    )


    print("="*60)
    print("REGRESSION EDA")
    print("="*60)


    print(
        "Shape:",
        df.shape
    )


    print(
        "\nMissing Values:"
    )

    print(
        df.isnull().sum()
    )


    print(
        "\nDuplicate Rows:",
        df.duplicated().sum()
    )


    print(
        "\nStatistics:"
    )

    print(
        df.describe()
    )



    plot_target_distribution(

        df,

        target_column,

        save_dir

    )



    plot_feature_target(

        df,

        target_column,

        save_dir

    )



    plot_correlation_heatmap(

        df,

        save_dir

    )
    plot_eda_summary(
    df,
    target_column,
    save_dir
    )



    print(
        "\nTarget Correlation:"
    )


    print(
        target_correlation(
            df,
            target_column
        )
    )