import glob
import os
from argparse import ArgumentParser
import json

import torch
import numpy as np
from pipeline.model import Word2GMM, Word2Vec, DotProduct
from pipeline.loss import EuclideanGMM
from pipeline.metrics import evaluate_similarity

def main(args):
    with open(args.dictionary, 'r') as f:
        dictionary = json.load(f)
    word_to_id = dictionary["dictionary"]
    with open(os.path.join(args.model_path, 'args.json'), 'r') as f:
        extra_args = json.load(f)
    ckpt_path = glob.glob(os.path.join(args.model_path, 'checkpoints/*.ckpt'))[0]
    print('\ncheckpoint:', ckpt_path)


    if extra_args["model"] == "Word2GMM":
        anchors_info = np.load(extra_args["anchors"])
        anchors = torch.from_numpy(anchors_info['embd'])
        anchors_indices = torch.LongTensor([word_to_id.get(w, 0) for w in anchors_info['word']])
        model = Word2GMM.load_from_checkpoint(ckpt_path, anchors_embd=anchors, 
                                            anchors_indices=anchors_indices,
                                            strict=True, map_location='cpu')
        distfunc = EuclideanGMM(reduction='none')
    elif extra_args["model"] == "Word2Vec":
        model = Word2Vec.load_from_checkpoint(ckpt_path, strict=True, map_location='cpu')
        def minus_dot_product(*args):
            result = DotProduct(*args)
            for i in args:
                result = result / torch.linalg.norm(i, dim=-1)
            return -result
        distfunc = minus_dot_product
    print(model.hparams)
    
    
    if args.similarity:
        print('\nSimilarity analysis:')
        files = glob.glob(args.similarity_dataset)
        results = evaluate_similarity(model, word_to_id, distfunc, files)
        csv_path = os.path.join(args.model_path, 'similarity.csv')
        results.to_csv(csv_path)
        print(results)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='lightning_logs/version_0')
    parser.add_argument('--dictionary', type=str, default='dictionary.json')
    parser.add_argument('--similarity', action='store_true')
    parser.add_argument('--similarity_dataset', type=str, default='corpus/evaluation_data/similarity_data/*.txt')

    args = parser.parse_args()
    main(args)