## Model and Data preparation for FFHQ experiments

The folder `FFHQ/prepare_models_data` contains the code to prepare models and data
for the FFHQ experiments.

### Convert StyleGAN2 weight from official checkpoints

First, you need to clone the official StyleGAN2 repositories, (https://github.com/NVlabs/stylegan2) as it is requires
for load official checkpoints.

For example, if you cloned repositories in ~/stylegan2 and
downloaded [stylegan2-ffhq-config-f.pkl](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl)
, you can convert it like this:

```bash
python convert_weight.py --repo ~/stylegan2 stylegan2-ffhq-config-f.pkl
```

The converted `stylegan2-ffhq-config-f.pt` file has to be moved to the directory `FFHQ/pretrained/stylegan2_pt`
where we load the pre-trained StyleGAN2 in the code.

### Data generation

Make sure you have downloaded the data (i.e., 10k pairs of latent variables and labels)
from [here](https://drive.google.com/file/d/1sFXqGpciLgqzjqMVCuWktH25ptLtXu_p/view?usp=sharing)
(originally from [StyleFlow](https://github.com/RameenAbdal/StyleFlow)) and unzip it to the `FFHQ` folder.

Generate 10k images from StyleGAN2 using the 10k labeled latent variables:

```bash
bash scripts/run_gen_from_latent.sh
```

The resulting pairs of generated images and labels will be used to train the FFHQ image classifier, and the generated
images only will be used to get the FID reference statistics.

### Training FFHQ image classifier

After `Data generation`, you first need to pre-train the image classifier on CelebA:

```bash
bash scripts/run_celeba_pretraining.sh
```

Note that the CelebA data is publicly available [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). In our code,
the images (`img_align_celeba.zip`) and labels (`list_attr_celeba.txt`) are stored in `./celeba`.

Then, you can fine-tune the above pre-trained image classifier on FFHQ:

```bash
bash scripts/run_image_clf.sh
```

The resulting FFHQ image classifier will be used to compute the conditional accuracy (ACC) score, and also be used to
annotate the MetFaces generated images.

### FID reference statistics

After `Data generation`, you can alculate the FID statistics for the FFHQ images:

```bash
bash scripts/run_calc_inception.sh
```

The resulting `inception_cifar10.pkl` will be used for computing FID scores.

