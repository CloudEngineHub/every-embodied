# 1X World Model Challenge

The rapid progress in video generation may make it possible to evaluate robot policies in a fully learned world model. A end-to-end learned simulator capable of simulating millions of robot environments will greatly accelerate the advancement of general robotics technology and provide useful insights for expanding data and computing capabilities.

To accelerate the progress of the robot learning simulator, we announced the 1X world model challenge, with the task of predicting the future first-person observation of EVE Android](https://www.1x.tech/androids/eve) from [EVE. We provided over 100 hours of vector-quantized image tokens and raw action data collected during manipulation of EVE, a baseline world model (GENIE style), and a frame-level MAGVIT2 autoencoder that compresses images into 16x16 tokens and decodes them back into images.

We hope that this dataset can assist robot researchers who wish to conduct experiments in a human body environment. A sufficiently powerful world model will allow anyone to access “neural-simulated EVE”. Evaluating the challenges is the ultimate goal, while we offer cash rewards for intermediate goals, such as well-fitting data (compression challenge) and realistic video generation (sampling challenge).

[Dataset on Huggingface ](https://huggingface.co/datasets/1x-technologies/worldmodel)

[ joined Discord](https://discord.gg/kk2HmvrQsN)

[ Phase 1 blog ](https://www.1x.tech/discover/1x-world-model), [ Phase 2 blog ](https://www.1x.tech/discover/1x-world-model-sampling-challenge)

Stay tuned for updates on the second phase of the 1X world model challenge!

See the [official 1X sampling-challenge page](https://www.1x.tech/discover/1x-world-model-sampling-challenge) for generated examples and phase updates. The textbook no longer embeds temporary GIF URLs from a third-party mirror.

## Challenge

Each example is a series of 16 first-person images from the robot, with a frame rate of 2 Hz (totaling 8 seconds). Your goal is to predict the next image based on the previous ones.

- **Compression Challenge ($10k prize)**: Predict the discrete distribution of tokens in the next image.
  - Standard: On our private test set, first implement a [[ temporal teacher forcing ](https://7mlcen.aitianhu6.top/c/67ccfd05-5854-800c-951e-b434719328e4#metric-details) loss below 8.0].
- **Sampling Challenge ($10k prize)**: Future prediction methods may not be limited to predicting the next logit. For example, you can use GANs, diffusion models, and MaskGIT to generate future images. The standard will be released later.
- **Evaluation Challenge (coming soon)**: Given a set of N policies $\pi_1, \pi_2, ... \pi_N$, where each policy $\pi_i(a_t|z_t)$ predicts action tokens based on image token predictions, can all policies be evaluated in the “world model” $p(z_{t+1}|z_t, a_t)$, and which policies rank best?

These challenges are largely inspired by the [ commavq compression challenge ](https://github.com/commaai/commavq). Please read the [ additional challenge details ](https://7mlcen.aitianhu6.top/c/67ccfd05-5854-800c-951e-b434719328e4#additional-challenge-details).

## Start

We require the use of `Python 3.10` or a later version. This code has been tested and passed on `Python 3.10.12`.

```
# 安装依赖并下载数据
./build.sh

# 激活Python环境
source venv/bin/activate
```



```
conda create -n your_environment_name python=3.10.12
conda activate your_environment_name

# Install dependencies
conda install -c conda-forge accelerate=0.30.1 torchvision=0.18.0 lpips=0.1.4 matplotlib tqdm wandb xformers=0.0.26.post1 wheel packaging ninja einops

# For transformers and torch (ensure compatibility with your CUDA version)
pip install transformers==4.41.0 lightning>2.3.1 git+https://github.com/janEbert/mup.git@fsdp-fix

pip install triton
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE python -m pip install flash-attn==2.5.8 --no-build-isolation
huggingface-cli download 1x-technologies/worldmodel --repo-type dataset --local-dir data
```

Required

export XFORMERS_FORCE_DISABLE_TRITON=1

## GENIE

This repository provides the implementation of the spatio-temporal transformer and MaskGIT sampler described in [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). Note that this implementation is trained only on video sequences, not on actions (although actions can be added easily by adding embeddings).

```
# 训练GENIE模型
python train.py --genie_config genie/configs/magvit_n32_h8_d256.json --output_dir data/genie_model --max_eval_steps 10

# 从训练好的模型生成帧
python genie/generate.py --checkpoint_dir data/genie_model/final_checkpt

# 可视化生成的帧
python visualize.py --token_dir data/genie_generated

# 评估训练好的模型
python genie/evaluate.py --checkpoint_dir data/genie_model/final_checkpt
```

### 1X GENIE Baseline

We have provided two pre-trained GENIE models, which are listed in [ on the ](https://7mlcen.aitianhu6.top/c/67ccfd05-5854-800c-951e-b434719328e4#leaderboard) ranking.

```
# 生成和可视化
output_dir='data/genie_baseline_generated'
for i in {0..240..10}; do
    python genie/generate.py --checkpoint_dir 1x-technologies/GENIE_138M \
        --output_dir $output_dir --example_ind $i --maskgit_steps 2 --temperature 0
    python visualize.py --token_dir $output_dir
    mv $output_dir/generated_offset0.gif $output_dir/example_$i.gif
    mv $output_dir/generated_comic_offset0.png $output_dir/example_$i.png
done

# 评估
python genie/evaluate.py --checkpoint_dir 1x-technologies/GENIE_138M --maskgit_steps 2
```

## Data Description

[ Please refer to the dataset card on Huggingface ](https://huggingface.co/datasets/1x-technologies/worldmodel).

The training dataset is stored in the `data/train_v1.1` directory.

## Participate in the challenge:

Please read the details of the additional challenge [ and ](https://7mlcen.aitianhu6.top/c/67ccfd05-5854-800c-951e-b434719328e4#additional-challenge-details) first to understand the rules.

Send the source code, build script, and some information about your method to [challenge@1x.tech](mailto:challenge@1x.tech). We will evaluate your submission on the dataset we hold and reply to you with the results via email.

Please send us the following:

- The username you choose (it can be your real name or alias, which will be bound to the email)
- The source code submitted as a.zip file
- The approximate number of FLOPs used during model training
- Any external data used during model training
- Your evaluation performance on the provided validation set (so we can roughly understand your expectations for the model)

After manually reviewing your code, we ran the evaluation in the 22.04 + CUDA 12.3 sandbox environment with the following command:

```
./build.sh # 安装所需的依赖和模型权重
./evaluate.py --val_data_dir <PATH-TO-HELD-OUT-DATA>  # 在保留的数据集上运行你的模型
```

## Additional Challenge Details

1. We have provided `magvit2.ckpt`, which is the weights of a [ MAGVIT2](https://github.com/TencentARC/Open-MAGVIT2) encoder/decoder. This encoder allows you to tokenize external data to improve metrics.
2. Unlike LLMs, loss metrics are not standardized because the vocabulary size of image tokens was changed when version 1.0 was released (July 8, 2024). We no longer compute cross-entropy loss for logits with 2^18 classes, but instead compute cross-entropy for predictions with 2x 2^9 classes and sum them up. This is because a large vocabulary size of 2^18 occupies a lot of memory when storing a logit tensor of shape `(B, 2^18, T, 16, 16)`. Therefore, the compression challenge considers the factorization PMF of model families in the form p(x1, x2) = p(x1)p(x2). For sampling and evaluation challenges, the factorization PMF is a necessary standard.
3. For the compression challenge, we deliberately choose to fix the factorization distribution p(x1, x2) = p(x1)p(x2) during training to evaluate retained data. Although models without factorization (e.g., p(x1, x2) = f(x1, x2)) should achieve lower cross-entropy on test data by utilizing the off-diagonal terms of the covariance of x1 and x2, we want to encourage solutions that achieve lower losses while maintaining the fixed factorization.
4. For the compression challenge, submissions can only use actions *before* the current prompt frame. Submissions can autoregressively predict subsequent actions to improve performance, but these actions will not be provided together with the prompt.
5. A simple nearest-neighbor retrieval + finding the next frame from the training set can achieve good loss and sampling results on the development and validation sets, as similar sequences exist in the training set. However, we explicitly prohibit this solution (the private test set will penalize such solutions).
6. If the challenge spirit is violated, we will not be able to reward individuals from countries subject to U.S. sanctions. We reserve the right not to award bonuses in the challenge.

### Metric Standard Details

The scenarios evaluated have different criteria, depending on the extent of the real context received by the model.
In decreasing order of context, these scenarios are:

- **Complete autoregressive**: The model receives a predetermined number of real frames and autoregressively predicts all remaining frames.
- **Temporal teacher forcing**: The model receives all real frames before the current frame, and autoregressively predicts all tokens in the current frame.
- **Complete teacher forcing**: The model receives all real tokens before the current frame, including the tokens in the current frame. Applicable only to causal LMs.

For example, consider predicting the final token of the video, which is located in the bottom-right corner of frame 15.
In each scenario, the context received by the model is as follows:

- Full autoregressive: The first $t$ 16x16 tokens are the true tokens from the first $t$ frames, and the remaining tokens are generated autoregressively. $0 < t < 15$ represents the number of predetermined prompt frames.
- Sequential teacher forcing: The first 15 16x16 tokens are the true tokens from the first 15 frames, and the rest are generated autoregressively.
- Full teacher forcing: All the first (16x16x16 - 1) tokens are true tokens.

The compression challenge uses the "timing teacher forcing" scenario.

## Rankings

These are the evaluation results on `data/val_v1.1`.

## Help us improve the challenge!

In addition to the world model challenges, we also hope to make the challenges and datasets more useful for *your* research questions. Do you want more data involving human interaction? More tasks requiring careful handling of critical safety operations, such as transporting hot coffee cups? More sophisticated tool usage? Robots collaborating with other robots? Robots dressing themselves in a mirror? Treat 1X as an operation team for obtaining high-quality human-like data, handling a wide variety of diverse scenarios.

Please send your data requirements (and why you believe they are important) to [challenge@1x.tech](mailto:challenge@1x.tech) by email. We will try our best to include them in future data releases. You can also discuss your data requirements with the community on [Discord](https://discord.gg/UMnzbTkw).

We also welcome donors to help us increase the bonuses.

## Citation

If you used this software or dataset in your work, please use the “Reference this repository” button on Github to cite it.

## Update Log

- v1.1 - Release the compression challenge standard; remove paused and discontinuous videos from the dataset; higher image cropping capabilities.
- v1.0 - More efficient MAGVIT2 tokenizer, mapping 16x16 (C=2^18) to 256x256 images, and provides original action data.
- v0.0.1 - Initial challenge release, using a 20x20 (C=1000) image tokenizer, mapping to 160x160 images.
