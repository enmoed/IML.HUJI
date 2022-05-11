from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "/Users/eitanmoed/Documents/Hebrew University/Classes/Year 2/Semester 2/Intro to Machine Learning/IML.HUJI/datasets/linearly_separable.npy"),
                 ("Linearly Inseparable",
                  "/Users/eitanmoed/Documents/Hebrew University/Classes/Year 2/Semester 2/Intro to Machine Learning/IML.HUJI/datasets/linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(perceptron: Perceptron, x_, y_):
            losses.append(perceptron.loss(X, y))

        per = Perceptron(callback=callback)
        per.fit(X, y)


        # Plot figure of loss as function of fitting iteration
        go.Figure([go.Scatter(x=[i for i in range(len(losses))],
                              y=losses, mode="markers+lines", name=n)]).show()



def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["/Users/eitanmoed/Documents/Hebrew University/Classes/Year 2/Semester 2/Intro to Machine Learning/IML.HUJI/datasets/gaussian1.npy",
              "/Users/eitanmoed/Documents/Hebrew University/Classes/Year 2/Semester 2/Intro to Machine Learning/IML.HUJI/datasets/gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes().fit(X, y)
        lda = LDA().fit(X, y)


        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Gaussian Naive"
                                                            " Bayes - "
                                                            "Accuracy: {0:.4f}".format(accuracy(gnb.predict(X), y)),
                                                            "LDA - Accuracy: "
                                                            "{0:.4f}".format(
                                                                accuracy(lda.predict(X), y))))

        # Add traces for data-points setting symbols and colors
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False,
                       marker=dict(color=gnb.predict(X),
                                   symbol=y,
                                   line=dict(color=gnb.predict(X),
                                             width=3))), 1, 1)
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False,
                       marker=dict(color=lda.predict(X), symbol=y,
                                   line=dict(color=lda.predict(X),
                                             width=3)), ), 1, 2)


        fig.update_xaxes(title_text="Feature 1")
        fig.update_yaxes(title_text="Feature 2")

        # Add `X` dots specifying fitted Gaussians' means



        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(3):
            fig.add_trace(go.Scatter(x=[gnb.mu_[i][0]], y=[gnb.mu_[i][1]],
                                     mode="markers",
                                     marker=dict(color="black", size=10,
                                                 symbol="x"),
                                     showlegend=False), 1, 1)
            fig.add_trace(go.Scatter(x=[lda.mu_[i][0]], y=[lda.mu_[i][1]], mode="markers",
                                     marker=dict(color="black", size=10, symbol="x"),
                                     showlegend=False), 1, 2)
            fig.add_trace(get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])), 1, 1)
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), 1, 2)
        fig.show()




if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
