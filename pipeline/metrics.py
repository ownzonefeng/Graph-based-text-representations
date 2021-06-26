import os

import torch
import numpy as np
import pandas as pd
from pipeline.utils import load_similarity_questions
from scipy import stats
from tqdm import tqdm


def evaluate_similarity(model, dictionary, distfunc, files):
    if isinstance(files, str):
        files = [files]
    results = {}
    all_corr = []
    each_len = []
    cnt = 0
    for f in tqdm(files):
        pairs, scores = load_similarity_questions(f)
        filename = os.path.basename(f)
        total_len = len(scores)
        cnt += total_len

        l, r = pairs[:, 0], pairs[:, 1]
        l_ids = torch.LongTensor([dictionary.get(w.lower(), 0) for w in l])
        r_ids = torch.LongTensor([dictionary.get(w.lower(), 0) for w in r])
        valid = (l_ids != 0) * (r_ids != 0)
        ratio = (torch.sum(valid) / len(scores)).item()
        l_ids = l_ids[valid]
        r_ids = r_ids[valid]
        scores = scores[valid]

        left = model(l_ids)
        right = model(r_ids)
        dists = distfunc(*left, *right)
        dists = - dists.numpy()

        corr = stats.stats.spearmanr(scores, dists)[0]
        all_corr.append(corr)
        each_len.append(len(scores))
        results[filename] = [round(corr, 4), total_len, len(scores), f'{ratio:2.2%}']
    
    all_corr = np.array(all_corr)
    each_len = np.array(each_len)
    overall = round(sum(all_corr * each_len / sum(each_len)), 4)
    results['Total'] = [overall, cnt, sum(each_len), f'{sum(each_len)/cnt:2.2%}']
    results = pd.DataFrame.from_dict(results, orient='index', columns=['Corr. Coef.', 'Total', 'Valid', 'Ratio'])
    return results