## Data preparation for AFHQ-Cats experiments

The folder `AFHQ/prepare_data` contains the code to prepare data for the AFHQ-Cats experiments.

### Pre-trained CLIP model

Since we apply the [CLIP model](https://github.com/openai/CLIP) to annotate the AFHQ-Cats data by designing proper
prompts that contain the controlling attributes, you first need to install it as a Python package:
```
pip install git+https://github.com/openai/CLIP.git
```
### Data generation

Generate 10k images and latent variables of StyleGAN2-ADA (including w and z):

```bash
bash scripts/run_gen_batch.sh
```

Use the pre-trained CLIP model to annotate the generated images:

```bash
bash scripts/run_clip_labeling.sh
```

The resulting pairs of latent variables and labels will be used to train latent classifiers.


