import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

#对初始选点进行优化
def kmeansplus(df, k):

    samplen=len(df['x'])

    #随机选择第一个中心点
    first = np.random.randint(0, samplen)
    centroids = {
        i: [np.zeros(2)]
        for i in range(k)
    }
    centroids[0][:] = df.loc[first, :]

    #选取后续的中心点
    juli = np.zeros(samplen)
    cen=0
    for i in range(k-1):
        cen=cen+1
        for j in range(cen):
            juli =juli + np.sqrt((df['x'] - centroids[j][0]) ** 2 + (df['y'] - centroids[j][1]) ** 2)
        rand = np.random.random() * sum(juli)
        leiji = 0
        for ii in range(samplen):
            leiji = leiji + juli[ii]
            if leiji >= rand:
                centroids[i+1][ :] = df.loc[ii, :]
                break
    return centroids


def assignment(df, centroids, colmap):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

def update(df, centroids):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids

def main():
    # step 0.0: generate source data
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })

    k = 3
    # centroids[i] = [x, y]
    centroids = kmeansplus(df, k)


    colmap = {0: 'r', 1: 'g', 2: 'b'}
    df = assignment(df, centroids, colmap)

    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()

    for i in range(10):
        key = cv2.waitKey()
        plt.close()

        closest_centroids = df['closest'].copy(deep=True)
        centroids = update(df, centroids)

        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()

        df = assignment(df, centroids, colmap)

        if closest_centroids.equals(df['closest']):
            break


if __name__ == '__main__':
    main()
