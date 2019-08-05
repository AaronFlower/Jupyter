import numpy as np
import matplotlib.pyplot as plt


def generate_2D_points(n=50, left_bottom=(0,0), right_top=(1,1)):
    """
    根据 left_bottom, right_top 确定的矩形，生成满足均匀分布的  n 个点.
    """
    x = np.random.uniform(left_bottom[0], right_top[0], n)
    y = np.random.uniform(left_bottom[1], right_top[1], n)
    return np.column_stack((x, y))


def get_data():
    area1 = generate_2D_points(40, (0.3, 0.3),  (0.5, 0.5))
    area2 = generate_2D_points(40, (0.5, 0.3),  (0.7, 0.6))
    area3 = generate_2D_points(50, (0.4, 0.4),  (0.6, 0.7))
    X = np.vstack((np.vstack((area1, area2)), area3))
    return X, (area1, area2, area3)


def get_k_centorids(X, k):
    """
    随机选择 k 个节点
    """
    m, _ = X.shape
    idx = np.random.permutation(m)
    return X[idx[0:k]]


def calc_dist(x1, x2):
    diff = x1 - x2
    return np.sum(np.square(diff))


def k_means(X, k=3, epochs=20):
    centroids = get_k_centorids(X, k)
    m, _ = X.shape
    min_loss = np.inf
    labels = np.zeros((m,))

    for epoch in range(epochs):
        # Assign label
        for i in range(m):
            min_dist = np.inf
            for j in range(k):
                dist = calc_dist(X[i], centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    labels[i] = j

        # Reassign centroids
        loss = 0.0
        for j in range(j):
            idx_j = labels == j
            centroids[j] = np.mean(X[idx_j], axis=0)
            loss += np.sum(np.sum(np.square(X[idx_j] - centroids[j]), axis=1))

        if loss >= min_loss:
            break
        else:
            min_loss = loss

        print('Epoch %d, loss %5.4f' % (epoch, loss))

    return centroids, labels


def main():
    k = 5
    X, areas = get_data()
    centroids, labels = k_means(X, k, 20)

    fig, axes = plt.subplots(1, 2)
    for i, area in enumerate(areas):
        axes[0].scatter(area[:, 0], area[:, 1], label='area' + str(i))
    axes[0].legend()

    for i in range(k):
        idx_i = labels == i
        cluster_i = X[idx_i]
        axes[1].scatter(cluster_i[:, 0], cluster_i[:, 1],
                        label='cluster' + str(i))

    axes[1].scatter(centroids[:, 0], centroids[:, 1], marker='X')
    axes[1].legend()
    plt.legend()
    plt.show()

    # fig.show()
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()


if __name__ == "__main__":
    main()
