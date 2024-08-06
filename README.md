# Devil in Shadow

## Install

- Build environment

```bash
conda env create -f environment.yml
```

- Download checkpoints

  You can learn about the program by [here](https://github.com/BradyFU/DVG-Face). The trained model is released on [here](https://www.dropbox.com/scl/fi/72fwflvt78h07d16lkphs/LightCNN128_epoch_15.pth.tar?rlkey=sg7np9gbpbhgkmmb0gqn674v5&dl=0).

  Pretrained LightCNN-29 model checkpoints that provided by [here](https://github.com/AlfredXiangWu/LightCNN).

- Download datasets

  In our experiment we use CASIA NIR-VIS 2.0 dataset for evaluation. Because we do not own the datasets,  you need to download them yourself. And you can refer to [CASIA NIR-VIS 2.0](https://github.com/bioidiap/bob.db.cbsr_nir_vis_2) for download.

  

## Train

```bash
python train.py
```

## Generate adversarial images

The generated image is saved in folder `./CASIA NIR-VIS 2.0_sample/CASIA_mtcnn_re`

```
python generate_images.py
```

## Test

```
python test.py
```

