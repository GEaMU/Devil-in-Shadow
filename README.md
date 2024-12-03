# Devil in Shadow: Attacking NIR-VIS Heterogeneous Face Recognition via Adversarial Shadow(IEEE TCSVT 2024)

This is the official implementation of the model proposed in the paper ([here]https://ieeexplore.ieee.org/document/10006826) with Pytorch.

## Comparison with SOTA methods
![图片1](/example/resultCASIA_and LAMP_cos.pdf "Comparison results of AP-ARA with our proposed method on CASIA NIR-VIS 2.0 datasets and LAMP-HQ datasets. The images in the first three rows are from the CASIA NIR-VIS 2.0 datasets, while the images in the last three rows are from the LAMP-HQ datasets. The number below the image indicates the similarity score of the same target face calculated by the recognition model.")
![图片1](/example/lightdifference.pdf "Schematic representation of the variation of light intensity.")

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

# Citations

