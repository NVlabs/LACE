## Data preparation for MetFaces experiments

The folder `MetFaces/prepare_data` contains the code to prepare data for the MetFaces experiments.

### Pre-trained FFHQ image classifier

Since we apply the FFHQ image classifier to annotate the MetFaces data, you first need to follow the
instruction [Training FFHQ image classifier](../../FFHQ/prepare_models_data/README.md) to get the pre-trained FFHQ image
classifier.

### Data generation

Generate 10k images and latent variables of StyleGAN2-ADA (including w and z):

```bash
bash scripts/run_gen_batch.sh
```

Use the pre-trained FFHQ image classifier to annotate the generated images:

```bash
bash scripts/run_metfaces_labeling.sh
```

The resulting pairs of latent variables and labels will be used to train latent classifiers.


