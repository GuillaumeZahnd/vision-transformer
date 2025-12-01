# Vision Transformer

## ALiBi: Attention with Linear Biase

ALiBi is a meaningful alternative to traditional sinusoidal position embeddings. See [Press et al., Train Short, Test Long (2022)](https://arxiv.org/pdf/2108.12409). From their abstract: *"ALiBi does not add positional embeddings
to word embeddings; instead, it biases query-key attention scores with a penalty that is proportional to their distance."*

### Patches

The input image is divided in square, potentially overlapping, patches. Patch extraction is carried out using a convolutional neural network. The `cls` token is appended before the first image patch.

![alibi_patches](https://github.com/user-attachments/assets/6cc3515b-f29d-4a98-af36-f6432ddbe4ba)

### Distances

The Euclidean distance is computed between all pairs of patches. Distances from a patch to itself are zero. Distances from the `cls` token to any image patch is zero, to represent the fact that the `cls` token can attend to any patch.

![alibi_distances](https://github.com/user-attachments/assets/ec393710-d63a-4413-b220-166762104780)

### Slope

A head-specific slope is computed, to express the distance with a different strength in function of the attention head.

![alibi_slopes](https://github.com/user-attachments/assets/c1f6446c-f965-4a36-8d5e-eafa99e94925)

### Biases

The head-specific biases are computed by multiplying the head-specific slope coefficient by the patch-wise distance and by `-1`.

![alibi_biases](https://github.com/user-attachments/assets/c8db27ee-02dd-497f-9542-3416b21b048d)

### Application

Unlike the traditional sinusoidal position embeddings approach, ALiBi [applies](https://github.com/GuillaumeZahnd/vision-transformer/blob/main/source/alibi_multi_head_self_attention.py#L49) positional encoding to the scaled attention logits, within the `softmax` operation.

### Alternatives

Other alternative approach for positional encoding include [Rotary Position Embedding (RoPE)](https://arxiv.org/pdf/2403.13298) with all its variants: vanilla 1D, Axial 2D, Mixed 2D, and the ["Golden gate RoPE"](https://jerryxio.ng/posts/nd-rope/), as well as [Lie Rotational Positional Encodings (LieRE)](https://arxiv.org/pdf/2406.10322v5).

## Python environment

```sh
python -m pip install --upgrade setuptools pip
mkdir .venv
pipenv install -d --python 3.12
```

## Hyperparameters

All hyperparameters are defined in the YAML files contained in the `config` folder and subfolders.

## Running the main training script

```sh
pipenv shell
python vit.py
```

## Monitor runs and logs with MLFlow Tracking

1. Start a local tracking server:

```sh
mlflow ui --port 8080 --backend-store-uri sqlite:///mlflow.db
```

2. Access the  interface via a web browser:

```sh
http://127.0.0.1:8080
```
