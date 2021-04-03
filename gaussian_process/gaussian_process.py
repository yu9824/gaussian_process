import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm


__all__ = [
    "GaussianProcess",
]


class GaussianProcess:
    def __init__(self):
        # np.arrayを0で割ったときのWarningを無視するためのコード
        np.seterr(divide='ignore', invalid='ignore')

    def _scale(self, x):
        x = np.array(x).reshape(-1, 1)
        scaler = StandardScaler().fit(x)
        scaled_x = scaler.transform(x)
        return scaled_x, scaler
    
    def _inverse_scale(self, scaler_y, mu, sigma):
        mu = np.array(mu).reshape(-1, 1)
        inversed_mu = scaler_y.inverse_transform(mu)
        inversed_sigma = sigma.reshape(-1, 1) * scaler_y.scale_
        return inversed_mu, inversed_sigma

    
    def fit(self, X, y, kernel = None, alpha = 1E-10, scale = True):
        """fit

        Parameters
        ----------
        X : list or something
            [description]
        y : list or something
            [description]
        kernel : callable, optional
            [description], by default None
        alpha : float, optional
            [description], by default 1E-10
        scale : bool, optional
            [description], by default True
        """
        # クラス変数化
        self.X = np.array(X).reshape(-1, 1)
        self.y = np.array(y).reshape(-1, 1)

        self.best_estimator = GaussianProcessRegressor(kernel = kernel, alpha=alpha)

        # あとでアクセスできるように
        self.best_kernel_ = kernel

        # 正規化 (必要な場合)
        if scale:
            scaled_y, self.scaler_y = self._scale(self.y)
        else:
            scaled_y = self.y
            self.scaler_y = None

        self.best_estimator.fit(self.X, scaled_y)

    
    def cross_validation(self, X, y, cv = 5, scoring = 'neg_mean_squared_error', scale = True, fit = True):
        """You can decide kernels by using cross validation. You don't have to do this if you've already decide which kernels do you use.

        Parameters
        ----------
        X : list or something
            [description]
        y : list or something
            [description]
        cv : int, optional
            [description], by default 5
        scoring : str, optional
            [description], by default 'neg_mean_squared_error'
        scale : bool, optional
            [description], by default True
        fit : bool, optional
            [description], by default True
        """
        # クラス変数化
        self.X = np.array(X).reshape(-1, 1)
        self.y = np.array(y).reshape(-1, 1)

        cv = min(cv, self.y.shape[0])
        
        # 目的変数を正規化
        if scale:
            scaled_y, self.scaler_y = self._scale(self.y)
        else:
            scaled_y = self.y
            self.scaler_y = None
        
        # カーネル
        kernels = [ConstantKernel() * DotProduct() + WhiteKernel(),
            ConstantKernel() * RBF() + WhiteKernel(),
            ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
            ConstantKernel() * RBF(np.ones(self.X.shape[1])) + WhiteKernel(),
            ConstantKernel() * RBF(np.ones(self.X.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
            ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
            ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
            ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
            ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
            ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
            ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()
        ]

        params = {
            'kernel':kernels
        }

        # GPRのインスタンス生成
        gpr = GaussianProcessRegressor(alpha=0)

        # kernel決定のためにcv
        gscv = GridSearchCV(gpr, params, cv = cv, scoring = scoring)
        gscv.fit(self.X, scaled_y)

        # 後でアクセスできるように
        self.best_score_ = gscv.best_score_
        self.results = pd.DataFrame.from_dict(gscv.cv_results_)

        # 最適なカーネルを使用
        self.best_kernel_ = gscv.best_params_['kernel']
        self.best_estimator = GaussianProcessRegressor(kernel = self.best_kernel_, alpha = 0)

        if fit:
            # fitさせる
            self.best_estimator.fit(X, scaled_y)

        
    def predict(self, plot_X):
        # クラス変数化
        self.plot_X = plot_X

        # predict
        self.mu, self.sigma = self.best_estimator.predict(self.plot_X, return_std = True)

        # scaleを戻す (必要な場合)
        if self.scaler_y is None:
            self.sigma = self.sigma.reshape(-1, 1)
        else:
            self.mu, self.sigma = self._inverse_scale(self.scaler_y, self.mu, self.sigma)
        return self.mu, self.sigma

    
    
    # 図生成関連
    def _formatting(self):
        # 図全体のフォーマット
        rcParams["font.family"] = "Helvetica"
        rcParams["font.size"] = 13
    
    
    def _plot_gp_results(self, offset = 0.05, ylabel = None):
        # offsetについて
        Xmin = min(self.plot_X)
        Xmax = max(self.plot_X)
        diff = Xmax - Xmin
        xlim = [Xmin - diff * offset, Xmax + diff * offset]

        # 範囲
        self.ax.set_xlim(xlim)

        # ylabel
        self.ax.set_ylabel(ylabel)

        # plot
        self.ax.plot(self.plot_X, self.mu, color = '#022C5E', label = 'mean', zorder = 1)
        self.ax.scatter(self.X, self.y, color = 'black', label = 'sample', zorder = 2)
        self.ax.fill_between(self.plot_X.squeeze(), (self.mu - 1.9600 * self.sigma).squeeze(), (self.mu + 1.9600 * self.sigma).squeeze(), zorder = 0, color = '#0572F7', label = '95 % confidence interval')

        return self.fig, self.ax

    
    def plot(self, offset = 0.05, xlabel = None, ylabel = None, figsize = (None, None)):
        """[summary]

        Parameters
        ----------
        offset : float, optional
            [description], by default 0.05
        xlabel : [type], optional
            [description], by default None
        ylabel : [type], optional
            [description], by default None
        figsize : tuple, optional
            [description], by default (None, None)

        Returns
        -------
        matplot.pyplot.figure, matplot.pyplot.axis
        """
        # フォーマットを綺麗に
        self._formatting()

        # figsizeについて
        if all(None is s for s in figsize) or figsize is None:
            figsize = rcParams["figure.figsize"]

        # 図の生成
        self.fig = plt.figure(facecolor = 'white', figsize = figsize)
        self.ax = self.fig.add_subplot(111)

        self._plot_gp_results(offset, ylabel)

        # xlabel
        self.ax.set_xlabel(xlabel)

        plt.tight_layout()

        return self.fig, self.ax

    
    def plot_with_acq(self, acquisition_function, offset = 0.05, xlabel = None, ylabel = None, figsize = (None, None)):
        """[summary]

        Parameters
        ----------
        acquisition_function : callable
            [description]
        offset : float, optional
            [description], by default 0.05
        xlabel : [type], optional
            [description], by default None
        ylabel : [type], optional
            [description], by default None
        figsize : tuple, optional
            [description], by default (None, None)

        Returns
        -------
        matplot.pyplot.figure, matplot.pyplot.axis, matplot.pyplot.axis
        """
        # フォーマットを綺麗に
        self._formatting()

        # 変数の定義
        af = acquisition_function
        if None in figsize or figsize is None:
            figsize = rcParams["figure.figsize"]

        #gridspecで互いにサイズの違うsubplotを作成
        gridspec_master = GridSpec(2, 1, height_ratios = [3, 1])
        print('反映されてる？')

        # 図の生成
        self.fig = plt.figure(facecolor = 'white', figsize = figsize)
        self.ax = self.fig.add_subplot(gridspec_master[0])

        self._plot_gp_results(offset, ylabel)

        # 獲得関数の図
        self.ax2 = self.fig.add_subplot(gridspec_master[1])

        # 獲得関数のy軸方向の値を得る
        acq, acq_name = af()

        # 獲得関数
        self.ax2.plot(af.plot_X, acq)

        # xlabel
        self.ax2.set_xlabel(xlabel)

        # ylabel
        self.ax2.set_ylabel(acq_name)

        plt.tight_layout()

        return self.fig, self.ax, self.ax2



    # 獲得関数関連
    class _acquisition_function:
        def __init__(self):
            pass

        def __call__(self): # np.ndarrayと獲得関数の名前をreturnするように書く．
            return np.ones(1), 'name'

        def get_optimum(self):
            index_opt = np.argmax(self.__call__()[0])
            X_opt = self.plot_X[index_opt][0]
            mu_opt = self.mu[index_opt]
            sigma_opt = self.sigma[index_opt]
            dict_opt = {
                'i':index_opt,
                'X':X_opt,
                'mu':mu_opt,
                'sigma':sigma_opt
            }
            return dict_opt


    class upper_confidence_bound(_acquisition_function):
        def __init__(self, gpr, plot_X, X, y):
            # 名前
            self.name = 'Upper Confidence Bound'

            # クラス変数化
            self.plot_X = plot_X

            self.mu, self.sigma = gpr.predict(self.plot_X, return_std=True)
            self.mu_sample = gpr.predict(X)

            n_sample = X.shape[0]
            mu_sample_opt = np.max(self.mu_sample)
            self.ucb = mu_sample_opt + np.sqrt(np.log(n_sample) / n_sample) * self.sigma
        
        def __call__(self):
            return self.ucb, self.name

        
            
    class expected_improvement(_acquisition_function):
        def __init__(self, gpr, plot_X, X, y, xi=0.01):
            # 名前
            self.name = 'Expected Improvement'

            # クラス変数化
            self.plot_X = plot_X

            self.mu, self.sigma = gpr.predict(self.plot_X, return_std=True)
            self.mu_sample = gpr.predict(X)
            self.sigma = self.sigma.reshape(-1, X.shape[1])

            mu_sample_opt = np.max(self.mu_sample)

            with np.errstate(divide='warn'):
                imp = self.mu - mu_sample_opt - xi
                Z = imp / self.sigma
                self.ei = imp * norm.cdf(Z) + self.sigma * norm.pdf(Z)
                self.ei[self.sigma == 0.0] = 0.0
        
        def __call__(self):
            return self.ei, self.name



if __name__ == '__main__':
    plot_X = np.linspace(0, 7, 701).reshape(-1, 1)
    X = np.array([0, 1, 2, 3, 5]).reshape(-1, 1)
    y = np.sin(X)
    xlabel = None
    ylabel = None
    
    # インスタンスの生成
    gp = gaussian_process()

    # best_estimatorを決め，fitまで終える
    # best_estimator = gp.cross_validation(X, y)
    best_estimator = gp.fit(X, y, kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel(), alpha=0)

    # predict
    mu, sigma = gp.predict(best_estimator, plot_X, scaler_y = gp.scaler_y)

    # 獲得関数の定義
    EI = gp.expected_improvement(best_estimator, plot_X, X, y)
    print(EI.get_optimum())

    # plot
    # gp.plot(plot_X, mu, sigma, xlabel=xlabel, ylabel = ylabel)
    gp.plot_with_acq(plot_X, X, y, mu, sigma, EI, xlabel = xlabel, ylabel = ylabel)


    plt.show()
    # print(gp.best_kernel_)
