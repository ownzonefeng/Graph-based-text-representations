import torch
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from sklearn import cluster
from datetime import datetime


def gaussian_prob_density(x, mu, sigma, normalized=False):    
    if normalized:
        k = mu.shape[-1]
        scaler = (2 * math.pi) ** (-k / 2)
        sigma_det = torch.prod(sigma, dim=-1) ** (-0.5)
    
    bias = (x - mu).unsqueeze(-2)
    sigma_inv = torch.diag_embed(1 / sigma)
    exp = torch.exp(-0.5 * bias @ sigma_inv @ bias.transpose(-1, -2))

    if normalized:
        return (scaler * sigma_det * exp)[..., 0, 0]
    else:
        return exp[..., 0, 0]


def load_similarity_questions(file_name):
    with open(file_name, "r", encoding="utf8") as file:
        dataset = file.readlines() 
    word_pairs = []
    scores = []
    for item in dataset:
        item = item.lower()
        A, B, score = item.split()
        word_pairs.append([A, B])
        scores.append(float(score))
    return np.array(word_pairs), np.array(scores)


def pick_anchors(graph='graph/wikidata_node8669_dim10.npz', num=128, method='DensitySampling'):
    wikidata_graph = np.load(graph)
    embd_matrix = wikidata_graph['embd_matrix']
    word = wikidata_graph['word']
    degree = wikidata_graph['degree']
    qid = wikidata_graph['qid']

    neigh = NearestNeighbors(n_neighbors=1).fit(embd_matrix)
    n_anchors = num
    ts = datetime.now().strftime('%d%m%y%H%M')
    name = f'graph/{ts}_{method}_anchor{num}.npz'


    if method == 'KMeans':
        kmeans = cluster.KMeans(n_clusters=n_anchors, n_init=50, max_iter=1000).fit(embd_matrix)
        ind = neigh.kneighbors(kmeans.cluster_centers_, n_neighbors=1, return_distance=False).flatten()
    elif method == 'HighDegree':
        ind = np.argsort(degree)[-1:-(n_anchors+1):-1]
    elif method == 'DegreeSampling':
        prob = np.clip(degree, 1, float('inf'))
        prob = prob / np.sum(prob)
        ind = np.random.choice(len(embd_matrix), size=n_anchors, replace=False, p=prob)
    elif method == 'DensitySampling':
        dist, _ = neigh.kneighbors(embd_matrix, n_neighbors=len(embd_matrix) // n_anchors + 1, return_distance=True)
        dist = np.sum(dist, axis=1)
        prob = (-dist + np.max(dist))
        prob = prob / np.sum(prob)
        ind = np.random.choice(len(embd_matrix), size=n_anchors, replace=False, p=prob)
    else:
        raise ValueError(method)

    np.savez(name, embd=embd_matrix[ind], word=word[ind], qid=qid[ind], degree=degree[ind])
    return name
