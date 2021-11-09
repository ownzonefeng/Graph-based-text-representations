# Graph-based text representations

This repository is the official implementation of the Master's thesis Graph-based text representations.

([Paper](https://github.com/ownzonefeng/Graph-based-text-representations/blob/main/open%20access/Master_thesis_Wentao_Feng.pdf), 
[Slides](https://github.com/ownzonefeng/Graph-based-text-representations/blob/main/open%20access/slides.pdf))
## Requirements

To install requirements:

```bash
conda env create -f environment.yml
```
or manually install:
- python >= 3.8.10
- pytorch >= 1.8.1
- pytorch-lightning >= 1.3.3
- torchtext >= 0.9.1
- spacy >= 3.0.6
- numpy
- scipy
- scikit-learn
- pandas

## Training

To train the model in the paper, run this command:

```bash
python train.py --model 'Word2GMM' --num_ns 5 --use_cache --batch_size 128 \
--lr 0.005 --rate_adjust 'StepLR' --target_coef 1 --n_dims 10 \
--n_gaussians 25 --center 1.2 --radius 0.2 --freeze_covar \
--anchors 'graph/wikidata_anchor128_dim10_DensitySampling.npz' --anchoring 'both' \
--precision 32  --max_epochs 50 --gpus 0,
```

## Evaluation

To evaluate my model on word similarity datasets, run:

```bash
python eval.py --model_path 'lightning_logs/pretrained' \
--similarity --similarity_dataset 'corpus/evaluation_data/similarity_data/*.txt'
```

## Pre-trained Model

You can download pretrained models [here](lightning_logs/pretrained/)

## Results

Our model achieves the following performance on:
### Word similarity
|                     |   Corr. Coef. |   Total |   Valid | Ratio   |
|:--------------------|--------------:|--------:|--------:|:--------|
| EN-MEN-TR-3k.txt    |        0.5041 |    3000 |    2315 | 77.17%  |
| EN-WS-353-ALL.txt   |        0.5212 |     353 |     323 | 91.50%  |
| EN-WS-353-SIM.txt   |        0.5455 |     203 |     185 | 91.13%  |
### Interpretability
Top-activated nodes for the word `water`:
- salmon
- lettuce
- fish
- tea
- rack

For the full results on word similarity and interpretability, please refer to the paper.

## Contributor
- [Wentao Feng](https://www.linkedin.com/in/wentaofeng)

## License
The content of this repository is released under the terms of the [MIT license](LICENSE).
