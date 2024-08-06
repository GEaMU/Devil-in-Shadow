# Devil in Shadow

## Install

- Build environment

```bash
conda env create -f environment.yml
```

- Download checkpoints

  Pretrained DVG-Face can be found [here](https://drive.google.com/file/d/0ByNaVHFekDPRWk5XUFRvTTRIVmc/view?resourcekey=0-1t3aWRoXB0wt9SPPWr-C6w).

  We use LightCNN-29 model checkpoints that provided by [here](https://drive.google.com/file/d/0ByNaVHFekDPRWk5XUFRvTTRIVmc/view?resourcekey=0-1t3aWRoXB0wt9SPPWr-C6w).

- Download datasets

  In our experiment we use CASIA NIR-VIS 2.0 dataset for evaluation. Because we do not own the datasets,  you need to download them yourself. And you can refer to [CASIA NIR-VIS 2.0](https://github.com/bioidiap/bob.db.cbsr_nir_vis_2) for download.

  

## Train

```bash
python train.py
```

## Generate adversarial images

```
python generate_images.py
```

## Test

```
python test.py
```

