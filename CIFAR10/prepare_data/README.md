## Data preparation for CIFAR-10 experiments

The folder `CFIAR10/prepare_data` contains the code to prepare data
for the CIFAR-10 experiments.

### Pre-trained image classifier

First, you need a pre-trained image classifier to annotate generated images. 

The image classifier
that we have used is publicly
available [here](https://mycuhk-my.sharepoint.com/personal/1155056070_link_cuhk_edu_hk/_layouts/15/guestaccess.aspx?folderid=0a380d1fece1443f0a2831b761df31905&authkey=Ac5yBC-FSE4oUJZ2Lsx7I5c). 
You can simply download the `densenet-bc-L190-k40` model and unzip it to the folder `CIFAR10/pretrained/classifiers/cifar10`, where our code will
load the checkpoint of the image classifier.

### Data generation

Generate 60k images and latent variables of StyleGAN2 (including w and z):

```bash
bash scripts/run_gen_batch.sh
```

Use the pre-trained image classifier to annotate the generated images:

```bash
bash scripts/run_cifar10_labeling.sh
```

The resulting pairs of latent variables (w and z) and labels will be used to train latent classifiers.

### FID reference statistics

After `Data generation`, you can calculate the FID statistics for real CIFAR-10 images:

```bash
bash scripts/run_calc_inception.sh
```

The resulting `inception_cifar10.pkl` will be used for computing FID scores.

