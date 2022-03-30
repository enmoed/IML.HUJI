from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, size=1000)
    func = UnivariateGaussian().fit(X)
    print(func.mu_)
    print(func.var_)


    # Question 2 - Empirically showing sample mean is consistent
    x_axis = [i for i in range(10,1001,10)]
    y_axis = []
    for i in range(len(x_axis)):
        y_axis.append(np.abs(10 - np.mean(X[:x_axis[i]])))

    go.Figure([go.Scatter(x=x_axis, y=y_axis, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{(5) Estimation of Expectation As Function Of Number Of Samples}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$|\mu-\hat\mu|$",
                  height=800)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    y_pdf = np.sort(X)
    x_pdf = func.pdf(y_pdf)


    go.Figure([go.Scatter(x=y_pdf, y=x_pdf, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{(5) Empirical PDF under the fitted model}$",
                  xaxis_title="r$\hat\mu$",
                  yaxis_title="PDF",
                  height=800)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.transpose(np.array([0, 0, 4, 0]))
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, sigma, size=1000)
    func = MultivariateGaussian().fit(X)
    print(func.mu_)
    print(func.cov_)

    # Question 5 - Likelihood evaluation
    f = np.linspace(-10, 10, 200)
    mu_5 = np.transpose(np.array(np.meshgrid(f, 0, f, 0))).reshape(-1, 4)
    log_func = lambda mu_: MultivariateGaussian.log_likelihood(mu_, sigma, X)

    log_like = np.transpose(np.apply_along_axis(log_func,  1, mu_5).reshape(
        200, 200))
    go.Figure(data=[go.Heatmap(x=f, y=f, z=log_like, type='heatmap')],
              layout=go.Layout(title="Heatmap of Log-Likelyhood "
                                     "Models", xaxis_title="F3",
                               yaxis_title="F1")).show()

    # Question 6 - Maximum likelihood
    ind = np.unravel_index(np.argmax(log_like, axis=None), log_like.shape)
    print(format(f[ind[0]], ".3f"))
    print(format(f[ind[1]], ".3f"))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
