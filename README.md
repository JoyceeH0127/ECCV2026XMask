# ECCV 2026 XMask

Repository of **Exclusivity-Guided Mask Learning for Semi-Supervised Crowd Instance Segmentation and Counting**, ECCV 2026 Spotloght🌟.

## Key contributions

- **Exclusion-Constrained Dual-Prompt SAM (EDP-SAM):** an automatic mask
generation framework for dense crowds. It combines head-point and
superpixel prompts with a Nearest Neighbor Exclusion Constraint (NNEC) to
reduce leakage across instances.
- **XMask (eXclusivity Mask):** an exclusivity-guided semi-supervised method
for dense crowd instance segmentation. It combines a discriminative mask
objective, Gaussian smoothing, and differentiable center sampling.
- **Mask Constraint Loss:** an instance-mask prior for semi-supervised crowd
counting. Reliable instance masks provide structured pseudo-labels with
spatial and shape information beyond sparse point annotations.



## Installation


```bash
conda create -n xmask python=3.9.23 pip
conda activate xmask

cd /path/to/XMask
python -m pip install -r requirements.txt
```



## Dataset downloads


| Dataset             | Download                                                                                                                                                                                                                              | Project support                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| ShanghaiTech Part A | [ShanghaiTechDataset mirror](https://github.com/desenzhou/ShanghaiTechDataset) · [Dropbox archive](https://www.dropbox.com/scl/fi/dkj5kulc9zj0rzesslck8/ShanghaiTech_Crowd_Counting_Dataset.zip?rlkey=ymbcj50ac04uvqn8p49j9af5f&dl=0) | Loader, `sha_{5,10,40}.yaml`, and labeled-subset protocols |
| UCF-QNRF            | [Official UCF CRCV page](https://www.crcv.ucf.edu/data/ucf-qnrf/) · [Official archive](https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip)                                                                                    | Loader, `ucf_{5,10,40}.yaml`, and labeled-subset protocols |
| JHU-CROWD++         | [Official JHU-CROWD++ page](http://www.crowd-counting.com/)                                                                                                                                                                           | Loader, `jhu_{5,10,40}.yaml`, and labeled-subset protocols |


Follow the license and terms published by each dataset. Official archives
must be converted to the layout below before they can be used by the current
loaders. **The crowd instance-mask annotations required for mask supervision
will be released separately**

## Data preparation and layout

Arrange processed images, point annotations, and instance masks as follows:

```text
datasets/
├── SHA/
│   ├── train_data/
│   │   ├── images/*.jpg
│   │   ├── new-anno/GT_{image_id}.npy
│   │   └── masks/{image_id}.npy
│   └── test_data/
│       ├── images/*.jpg
│       ├── new-anno/GT_{image_id}.npy
│       └── masks/{image_id}.npy
├── UCF/
│   ├── train_data/
│   │   ├── images/*.jpg
│   │   ├── new-anno/{image_id}_ann.npy
│   │   └── masks/{image_id}.npy
│   └── test_data/
│       ├── images/*.jpg
│       ├── new-anno/{image_id}_ann.npy
│       └── masks/{image_id}.npy
└── JHU/
    ├── train_data/
    │   ├── images/*.jpg
    │   ├── new-anno/{image_id}.npy
    │   └── masks/{image_id}.npy
    └── test_data/
        ├── images/*.jpg
        ├── new-anno/{image_id}.npy
        └── masks/{image_id}.npy
```

Processed datasets can be stored elsewhere and linked into the repository:

```bash
ln -sfn /path/to/processed/SHA datasets/SHA
ln -sfn /path/to/processed/UCF datasets/UCF
ln -sfn /path/to/processed/JHU datasets/JHU
```


## Training

Run one configuration from the repository root:

```bash
# ShanghaiTech Part A with 5% labeled data
python train.py --config configs/sha_5.yaml

# UCF-QNRF with 5% labeled data
python train.py --config configs/ucf_5.yaml

# JHU-CROWD++ with 5% labeled data
python train.py --config configs/jhu_5.yaml
```

The helper scripts run one configuration or all nine configurations:

```bash
bash scripts/run_one.sh configs/sha_5.yaml
bash scripts/run_all.sh
```

Training results are written to `outputs/<tag>/`.

Resume a run with:

```bash
python train.py \
  --config configs/sha_5.yaml \
  --resume outputs/sha-L5/ckpt_epoch_1499.pth
```

Configuration values can be overridden with `--opts KEY VALUE` pairs. Paths
remain relative to the repository root:

```bash
python train.py \
  --config configs/sha_5.yaml \
  --opts data.data_path /path/to/SHA train.epochs 2200
```



## Evaluation



### Complete evaluation

Run counting and instance-segmentation evaluation in one model pass:

```bash
python eval.py \
  --config configs/sha_5.yaml \
  --checkpoint outputs/sha-L5/ckpt_best.pth
```



### Counting only

```bash
python eval_count.py \
  --config configs/sha_5.yaml \
  --checkpoint outputs/sha-L5/ckpt_best.pth
```

Counting evaluation treats the number of low-resolution positions and reports MAE and RMSE.

### Instance segmentation only

```bash
python eval_seg.py \
  --config configs/sha_5.yaml \
  --checkpoint outputs/sha-L5/ckpt_best.pth
```

Instance segmentation reports Mean IoU: the sum of Hungarian-matched IoUs divided by the total number of ground-truth instances.

## Project structure

```text
XMask/
├── train.py                    # unified three-phase training
├── eval.py                     # complete count + segmentation evaluation
├── eval_count.py               # MAE and RMSE
├── eval_seg.py                 # Mean IoU C
├── configs/                    # dataset and labeled-ratio configurations
├── protocols/                  # labeled subsets for semi-supervised training
├── scripts/                    # single and batch training scripts
├── datasets/                   # processed datasets or symbolic links
└── xmask/
    ├── data/                   # SHA, UCF-QNRF, and JHU-CROWD++ loaders
    ├── models/                 # VGG16-BN, density decoder, and MaskHead
    ├── losses/                 # P2R, discriminative, and mask constraints
    ├── pseudo/                 # online pseudo-instance generation
    ├── evaluation.py           # shared evaluation metrics
    ├── config.py
    └── utils.py
```



## Citation

```bibtex
@misc{huang2026exclusivityguidedmasklearningsemisupervised,
  title={Exclusivity-Guided Mask Learning for Semi-Supervised Crowd Instance Segmentation and Counting},
  author={Jiyang Huang and Hongru Chen and Wei Lin and Jia Wan and Antoni B. Chan},
  year={2026},
  eprint={2603.16241},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.16241}
}
```



## License

This project is released under the [MIT License](LICENSE).
