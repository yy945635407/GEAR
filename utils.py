import matplotlib.pyplot as plt
import sklearn
import numpy as np
from scipy import signal
from tqdm import tqdm
from scipy.fft import fft
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsRegressor

def generate_x(t, f1, alpha, r, T=1024):
    '''
    giving time duration t, parameters f1, alpha and r, generate the signal x

    Parameters:
    ----------
    t: time indices with shape (T,)
    f1: f1 in formula, scalar
    alpha: alpha in formula, scalar
    r: r in formula, scalar
    N: constant in formula, scalar
    T: constant in formula, scalar

    Returns:
    -------
    x: generated result with shape (1, T)
    '''
    # ensure the maximum frequency is lower than Nyquest frequency
    N = int(T/(2*f1)) - 1 
    # weights with shape(N, 1)
    a = np.array([(1 + np.power(-1, n) * r)/(np.power(n, alpha)) 
                    for n in range(1, N+1)])[:, np.newaxis] 
    # cosines with shape (N, T) 
    cosines = np.array([np.cos(2 * np.pi * n * f1 * t) for n in range(1, N+1)]) 
    # window with shape(1, T)
    window = np.hanning(T)[np.newaxis, :] 
    # element-wise multiply and sum at the first dimension
    # (sumation of N components)
    x = np.sum(a * cosines * window, axis=0) 
    
    return x

def synthesis_data(T, n_alpha, n_r, lb_alpha, ub_alpha, lb_r, ub_r, lb_f1, 
                   ub_f1, log_f1=False, avoid_zero=True, random_f1=True, 
                   seed=0):
    '''
    Synthesis data and the corresponding labels

    Parameters:
    ---
    T: length of each signal
    n_alpha: number of choices of alpha
    n_r: number of choices of r
    lb_alpha: lower bound of alpha
    ub_alpha: upper bound of alpha
    lb_r: lower bound of r
    ub_r: upper bound of r
    lb_f1: lower bound of f1
    up_f1: lower bound of f1
    log_f1: whether to use logspace or linearspace of f1
    avoid_zero: whether to avoid zeros in parameters
    random_f1: whehter to chose f1 randomely or choose it in range. If True,
    f1 is chosen in range randomely for each signal; if False, f1 is chosen 
    similarly as alpha and r
    seed: number of random seed of numpy

    Returns:
    ---
    X: sythesis X with shape(n_samples, T) where n_samples = n_alpha * n_r * 
    n_f1
    Theta: params with shape(n_samples, 3)
    Params: list of names of params
    '''
    np.random.seed(seed)
    # n_alpha values in [lb_r, ub_r)
    alpha_range = np.arange(lb_alpha , ub_alpha, (ub_alpha-lb_alpha)/n_alpha) 
    # n_r values in [lb_r, ub_r)
    r_range = np.arange(lb_r, ub_r, (ub_r-lb_r)/n_r) 
    # f1 range as integers
    f1_range = range(lb_f1, ub_f1) 
    t_range = np.arange(T)/T
    X = []
    Theta = []
    if random_f1:
        # add a progress bar
        with tqdm(total=n_alpha * n_r) as bar:
            # traverse all the parameters
            for alpha in alpha_range:
                for r in r_range:
                    # f1 is an integer sampled randomly in range
                    f1 = np.random.choice(f1_range, 1)[0] 
                    if avoid_zero:
                        # To avoid zeros when using relative metric
                        # (result in 0s)
                        if 0 in [alpha, r, f1]:
                            bar.update()
                            continue
                    x = generate_x(t_range, f1, alpha, r, T)
                    Theta.append([f1, alpha, r])
                    X.append(x)
                    bar.update()
    else:
        # f1 can only be integers, so the upper bound and lower bound decides 
        # the number of its values
         with tqdm(total=n_alpha * n_r * len(f1_range)) as bar:
            for f1 in f1_range:
                for alpha in alpha_range:
                    for r in r_range:
                        if avoid_zero:
                            if 0 in [alpha, r, f1]: 
                                bar.update()
                                continue
                        x = generate_x(t_range, f1, alpha, r, T)
                        Theta.append([f1, alpha, r])
                        X.append(x)
                        bar.update()
    Params = ["f1", "alpha", "r"]
    Theta = np.array(Theta)
    X = np.array(X)
    return X, Theta, Params

def plot_sampled_signal(X, Theta, N_sample):
    '''
    uniformly sample data and plot each original signal, fourier transform and
    fourier transform in three subplots respectively
    
    Parameters:
    ---
    X: data with shape (num_signals, len(signal))
    Theta: parameters theta with shape (num_signals, 3)
    N_sample: int value, number of samples shown in each plot
    
    '''
    # calculate number of rows and columns of subplots
    # to make it looks similar to a square plot
    n_row = int(np.ceil(np.sqrt(N_sample)))
    n_col = int(np.ceil(N_sample / n_row))
    # calculate gap between each sampled signal
    n_step = int(np.floor(X.shape[0] / N_sample))
    # create three subplots
    fig_sig, ax_sig = plt.subplots(n_row, n_col, sharex=True, sharey=False, 
                                   figsize=(25, 20))
    fig_fx, ax_fx = plt.subplots(n_row, n_col, sharex=True, sharey=False, 
                                 figsize=(25, 20))
    fig_fxlg, ax_fxlg = plt.subplots(n_row, n_col, sharex=True, sharey=False, 
                                     figsize=(25, 20))

    # title three subplots
    fig_sig.suptitle("Original Signal(f1, r, alpha)")
    fig_fx.suptitle("Fourier Transform(f1, r, alpha)")
    fig_fxlg.suptitle("Fourier Transform in Log Scale(f1, r, alpha)")

    for i in range(n_row):
        for j in range(n_col):
            index = (i * n_row + j) * n_step
            if index >= X.shape[0]:
                continue
            x = X[index, :]
            theta = Theta[index, :]
            fx = fft(x)

            # original signal 
            ax_sig[i, j].plot(x)
            ax_sig[i, j].set_title("{f1}, {r:.2}, {alpha:.2}"
                                    .format(f1=theta[0], alpha=theta[1], 
                                            r=theta[2]), c='g', fontsize=10)
            # Fourier tranform
            ax_fx[i, j].plot(abs(fx))
            ax_fx[i, j].set_title("{f1}, {r:.2}, {alpha:.2}"
                                    .format(f1=theta[0], alpha=theta[1], 
                                            r=theta[2]), c='g', fontsize=10)
            # Fourier transform in log scale
            ax_fxlg[i, j].plot(abs(fx))
            ax_fxlg[i, j].set_title("{f1}, {r:.2}, {alpha:.2}"
                                    .format(f1=theta[0], alpha=theta[1], 
                                            r=theta[2]), c='g', fontsize=10)
            # log scale
            ax_fxlg[i, j].set_xscale("log")

    plt.show()

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_3d_scatter(all_embeddings, y, params, N_neighbors=100, N_components=3, 
                    s=3):
    '''
    Plot all embedding models' 3d Isomap results

    Parameters:
    -----------
    all_embeddings: dict, containing all kinds of embedding results, key and 
    value pair is (embedding_name(str): embedding(np.array))
    y: np.array, true values of parameters with shape (n_samples, n_params)
    params: list, list of names of parameters
    N_neighbors: number of neighbors taken into account when using Isomap
    N_components: dimension of output of Isomap, default to 3 (3d plot)
    s: size of points in plot

    Returns:
    --------
    all_isomap: dict, containing isomap results corresponding to all_embeddings
    '''
    all_isomap = {}
    for emb_name in all_embeddings.keys():
        embeddings = all_embeddings[emb_name]
        emb_iso = Isomap(n_neighbors=N_neighbors, n_components=N_components)\
                  .fit_transform(embeddings)
        # add to dict
        all_isomap[emb_name] = emb_iso
        print("----Results of " + str(emb_name) + " embeddings-----")
        fig = plt.figure(figsize=plt.figaspect(0.3))
        # one line 3 plots for one embedding model
        for i in range(y.shape[1]):
            ax = fig.add_subplot(1, 3, i+1, projection="3d")
            ax.scatter(emb_iso[:, 0], emb_iso[:, 1], emb_iso[:, 2], 
                       s=s, c=y[:,i], cmap='coolwarm')
            ax.set_title(params[i])
            # hide labels for three axises
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            # make three axis at the same scale
            set_axes_equal(ax)

        plt.show()
        
    return all_isomap

def cal_knn_and_plot_score(embeddings, y, params, N_neighbors=6, fixed=True):
    '''
    calculate knn and plot the relative score of each model
    ---
    Parameters:
    -----------
    embeddings: dict of all embeddings with mapping 'model_name': embeddings 
    of model(n_samples, n_embedding)
    y: labels with shape(n_samples, n_freedom)
    params: list of names of params (with length of n_freedom)
    models: list of string containing all models' names
    N_neighbors: parameter of knn algorith, number of neighbors taken into 
    account

    Returns:
    --------
    scores_dict: dict of relative scores corresponding to all the models in embeddings
                 with keys as names of params, and values as array with shape 
                 (n_models, n_samples)

    '''
    visual_scores = [3, 2, 1, 1/2, 1/3]
    visual_texts = ["3", "2", "1", "1/2", "1/3"]
    names = []
    scores_dict = {}
    # generate points aroung 1, 2, 3, 4 and so on
    for i in range(len(embeddings)):
        # with lenth 0.1
        names += list(np.random.uniform(low=i+1-0.05, high=i+1+0.05
                                        , size=(y.shape[0],))) # [i] * y.shape[0]
    models = list(embeddings.keys())
    scores = []
    for embedding in embeddings.values():
        if fixed:
            knn_model = KNeighborsRegressor(n_neighbors=N_neighbors)
        else:
            # adaptive neighbor nums
            knn_model = KNeighborsRegressor(
                n_neighbors=min(2 * embedding.shape[1], embedding.shape[0])) 
        knn_model.fit(embedding, y)
        knn_prediction = knn_model.predict(embedding)
        score = knn_prediction/y
        scores.append(score)
    scores = np.array(scores)
    for dim_free in range(y.shape[1]):
        scores_dict[params[dim_free]] = scores[:, :, dim_free]
        plt.figure()
        plt.scatter(names, scores[:, :, dim_free], c=names, 
                    cmap='coolwarm', s=0.3)
        plt.xticks([i+1 for i in range(len(models))], models)
        plt.xlabel("embedding models")
        plt.ylabel("relative score")
        # using log scale for score
        plt.yscale("function", functions=
                   (lambda x: np.log10(x), lambda x: np.power(10, x))) 
        # only 'ticks' after 'scale' make sense
        plt.yticks(visual_scores, visual_texts)
        plt.ylim([1/3, 3])
        # add dotted grid
        plt.grid(linestyle="--")
        plt.title("Relative scores of different models on param " 
                  + params[dim_free])
    # On average
    # plt.figure()
    # plt.scatter(names, np.mean(scores, axis=-1), c=names, cmap='coolwarm', s=5)
    # plt.xticks([i for i in range(len(models))], models)
    # plt.xlabel("embedding models")
    # plt.ylabel("relative score")
    # plt.title("Relative scores of different models average ")
    return scores, scores_dict
