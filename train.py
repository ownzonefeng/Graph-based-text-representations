import glob
import os
from argparse import ArgumentParser
import json

import torch
import numpy as np
from torch.utils.data import DataLoader
from pipeline.data import SkipGram
from pipeline.model import Word2GMM, Word2Vec
from pipeline.utils import pick_anchors
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin


def main(args):
    skip_gram_dataset = SkipGram(glob.glob(args.data_path), num_ns=args.num_ns, use_cache=args.use_cache, reload=args.reload)
    dataloader = DataLoader(skip_gram_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)

    if args.anchors == 'OnTheFly' and args.model == 'Word2GMM':
        args.anchors = pick_anchors(args.graph, args.n_anchors, args.method)
        print('Anchors saved:', args.anchors)
    
    if args.model == 'Word2GMM':
        anchors_info = np.load(args.anchors)
        anchors = torch.from_numpy(anchors_info['embd'])
        anchors_indices = torch.LongTensor(skip_gram_dataset(anchors_info['word']))

    n_words = skip_gram_dataset.dict_size
    if args.model == 'Word2GMM':
        model = Word2GMM(n_words, anchors_embd=anchors, anchors_indices=anchors_indices, **vars(args))
    elif args.model == 'Word2Vec':
        model = Word2Vec(n_words, **vars(args))
    else:
        raise ValueError(args.model)

    early_stop_callback = EarlyStopping(monitor='train_loss', min_delta=0.001, patience=10)
    plugins_setting = DDPPlugin(find_unused_parameters=False) if args.accelerator == 'ddp' else None
    trainer = Trainer.from_argparse_args(args, callbacks=[early_stop_callback], plugins=plugins_setting)
    trainer.fit(model, train_dataloader=dataloader)

    with open(os.path.join(trainer.log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(trainer.log_dir, 'dictionary.json'), 'w') as f:
        json.dump(skip_gram_dataset.info, f, indent=2)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='Word2GMM')
    parser.add_argument('--data_path', type=str, default='corpus/training_data/*.txt')
    parser.add_argument('--num_ns', type=int, default=5)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--anchors', type=str, default='graph/wikidata_anchor128_dim10_DensitySampling.npz')
    parser.add_argument('--graph', type=str, default='graph/wikidata_node8669_dim10.npz')
    parser.add_argument('--n_anchors', type=int, default=128)
    parser.add_argument('--method', type=str, default='DensitySampling')

    parser = Word2GMM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    main(args)
