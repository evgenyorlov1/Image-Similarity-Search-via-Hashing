dump_file = 'hash.dat'


def pca(data, R=2):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=R, svd_solver='full')
    pca.fit(data)
    data = pca.transform(data)
    evr = sum(pca.explained_variance_ratio_)

    return data, evr


def plot(data):
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    plt.scatter(data[1:45, 0], data[1:45, 1], c='g')
    plt.scatter(data[46:90, 0], data[46:90, 1], c='b')
    plt.scatter(data[91:, 0], data[91:, 1], c='r')
    green_sc = mlines.Line2D([], [], color='g', label='Scene 1')
    blue_sc = mlines.Line2D([], [], color='b', label='Scene 2')
    red_sc = mlines.Line2D([], [], color='r', label='Scene 3')
    plt.legend(handles=[green_sc, blue_sc, red_sc])

    plt.show()


def run():
    import numpy as np

    data = np.load(dump_file)
    data, evr = pca(data)
    print evr
    plot(data)


run()