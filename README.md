## FIN Llamma2 - Llama2 with Stochastic Gradient Descent (SGD) for Classification

<p align="center">
  <img src="assets/FIN_TECH_LLAMA.png" width="300" height="300" alt="Fin Tech Llama">
</p>

Have you ever thought about training a llama to work on Wall Street? 

Research has shown that very small LLMs can have surprisingly strong performance if you make the domain narrow enough (ref: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) paper).

However, when it comes to classification problems, the commonly used Adam Optimizer for training Language Models might not be the best choice: 
1. Ref: [Towards Theoretically Understanding Why SGD Generalizes Better Than ADAM in Deep Learning](https://arXiv:2010.05627v2) paper.
2. Ref: [Improving Generalization Performance by Switching from Adam to SGD](https://arXiv:1712.07628v1) paper.
3. Ref: [The Marginal Value of Adaptive Gradient Methods in Machine Learning Paper](https://arXiv:1705.08292v2]) paper.
4. Ref: [A Bounded Scheduling Method for Adaptive Gradient Methods](https://www.mdpi.com/2076-3417/9/17/3569) paper.

As you can see in the image below, from [A Bounded Scheduling Method for Adaptive Gradient Methods](https://www.mdpi.com/2076-3417/9/17/3569), Adam converges rapidly to a “sharp minima” while SGD converges to a “flat minima” and performs better on the test data. This improved generalization makes SGD a better choice for classification problems.

<p align="center">
  <img src="assets/A Bounded Scheduling Method for Adaptive Gradient Methods.png" height="300" alt="Source: A Bounded Scheduling Method for Adaptive Gradient Methods">
</p>

I've learned through experimentation that when the foundation model was trained to a "sharp minima" with AdamW, it was also more challenging to train the model on an ongoing basis. Since SGD is incrementally Stochastic in nature, models can be fine-tuned with a flat learning rate on an ongoing basis to market drift daily or weekly. 

One additional advantage of SGD is it has a much smaller memory footprint than Adam. It enables you to train much larger models than you could train on the same GPU resources with Adam. With a single 24 GB RTX 4090, you can train a foundational 1.3 Billion Parameter Model on 50 Billion Tokens of data in about a month (It would be much faster with a rack of GPUs purring). 

Thank you to [Andre Karpathy](https://github.com/karpathy/llama2.c) for the initial training script and model card, which was modified for this project. 

This repo will become a "full-stack" train + inference solution for Llama 2 LLM in Python and Pytorch, with SGC and a focus on minimalism and simplicity. 


## Training

First, navigate to the folder where you keep your projects and clone this repository to this folder:

```bash
git clone https://github.com/kevinmeehanofficial/FIN_LLAMA.git
```

Then, open the repository folder:

```bash
cd FIN_LLAMA
```

You will need to create two .bin files that contain tokenized data. I will be releasing a tokenizer I built for this in the near future, but just make sure that the dtype=np.uint16. Store the val.bin and train.bin files in a subdirectory named "data".

Once data is cleaned, tokenized, and prepared, then train our model:

```bash
python train.py
```

**brief training guide**. See the train.py script for more exotic launches and hyperparameter overrides. Here is a brief guide to how to set the parameters. Look at the table at the very end of the [Chinchilla paper](https://arxiv.org/abs/2203.15556) to get a sense of how the Transformer parameters (dim, n_layers, n_heads) grow or shrink together. Extrapolate/interpolate this pattern to get bigger or smaller transformers. Set the max context length however you wish, depending on the problem: this should be the max number of tokens that matter to predict the next token. E.g. Llama 2 uses 2048. Next, you want the _total_ batch size per update (printed by the script as "tokens per iteration will be:") to be somewhere around 100K tokens for medium-sized applications. For tiny applications it could be lower, for large training (e.g. GPTs/LLamas) it is usually ~0.5M, or even more. You get there by first maxing out the batch_size to whatever your system allows (e.g. mine was 16 in a recent run because after that my GPU runs out of memory), and then you want to increase gradient_accumulation_steps to be as high as necessary to reach the total batch size of ~100K. Finally, you want to tune your learning_rate (LR). You want this to be as high as your training allows. Very small networks can get away with a large LR (e.g. 1e-3 or even higher). Large networks need lower LRs. 3e-4 is a safe choice in most medium-sized applications, but can be too low for small networks, so try to increase it! Finally, max_iters is the length of training. Play with different settings. I mostly only ever tune these parameters and leave most of the others unchanged. Here is an example of how I trained the 110M model, which I don't think is anywhere near optimal, but looked sensible to me: dim 768, n_layers 12, n_heads 12 (so size of each head is 768 / 12 = 64 channels), seq len of 1024, batch size 16 (this is the most that fit my A100 40GB GPU), gradient_accumulation_steps = 8 was needed to get total tokens batch size to be 16 batch size * 1024 tokens in sequence * 8 grad_accum = 131,072 tokens per update. Good. Learning rate 4e-4 (probably a little too low). max_iters 200K (probably a bit too high). Dropout 0.1, as that usually helps a bit at medium size. That was it. I ran using Distributed Data Parallel (DDP) on 4 GPUs on my cloud machine, training took ~day or so.

I will also release a Redis interface script for cloud deployment of the trained model. 

## custom tokenizers

With this script, you can use any tokenizer you want. I used a customized version of Tiktoken, which I optimized for financial data (more on this in the near future). As long as you create a train.bin and val.bin file with tokens in dtype=np.uint16 format, and store them in a subdirectory called data; feel free to use whatever you want. Sentence piece is commonly used for Language Models aswell.

## unsorted todos

- update the training guide based on my experiences
- upload custom tokenizer
- upload interface script for a completed model
- clean up/ simplify train script
- optimize everything

## License

MIT
