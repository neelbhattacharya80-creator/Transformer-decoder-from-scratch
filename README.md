# Transformer From Scratch

## Overview

This project is a decoder-only Transformer implemented almost entirely from scratch to better understand how modern language models work at a low level.

PyTorch is only used for automatic differentiation, tensor operations, and GPU acceleration. Core components including multi-head self-attention, RoPE positional encoding, AdamW, RMS Norm, dropout, Swiglu, AdamW, cross-entropy loss, KV caching, batching, and the training loop—are implemented manually. The current model contains approximately **200 million trainable parameters** trained on 4 billion fineweb-edu tokens following scaling laws.

## Learning Outcomes

The goal was to produce a Language Model similar to GPT2 without relying on high-level abstractions.
Building from first principles required a complete understanding of the architechture and pipeline and the mathematical
building blocks along with extensive focus towards optimisation to ensure the model trained relatively fast to be trained to a reasonable level.
Initially the model used LayerNorm and FFNs which were later updated to RMS norm,Swiglu along with residual scaling and Mixed Precision.

Training included Custom made AdamW along with warmup and cosine decay. First run a 95M model was trained on 1.2 B tokens 
for 3 epochs which stalled around 3.8 a second training run with a fresh 2B dataset was run for 15k steps however a lr of 5e-7 
was used which resulted in no progress atall. Finally 200M model was trained on 4B tokens with a lr of 3e-4 reaching convergence around 
~3.42(with dropout) and between 3.3 without dropout.

Inference was unusually slow at first due to the model being loaded in the cpu. Output quality was extremely repetitive with no coherence
before implementing repetition penalty which seemed to solve the problem very well. PE bugs during inference also significantly reduced
context.


## Training Progress

Model Trained on 4 Billion fineweb-edu tokens following optimal scaling 
with 10% dropout model stalled at ~3.4 however on a test run done with 0 dropout model performed significantly better being in the 
3.28-3.38 range
Loss progression was very fast and gradually decreased after crossing 4 at which point it gradually converged around 3.42(with dropout)

## Features

* Decoder-only Transformer
* Multi-head self-attention
* Rotary Positional Embeddings (RoPE)
* KV cache for efficient autoregressive inference
* Pre-Norm architecture
* Swiglu
* RMS Norm
* Custom AdamW optimizer
* Gradient accumulation
* Gradient clipping
* Cosine learning-rate decay with warmup
* Weight tying
* GPT-2 tokenization via tiktoken
* Top-k sampling for text generation

## Sample Output
Total trainable parameters: 205,629,440
Starting conversation: 

You:The US has been the leading figure in terms of AI

GPT:  adoption.
In March, Google announced a $400 million investment into artificial intelligence (AI) by Google Inc., which will create a new company called Google Insights that would help developers find high-quality and meaningful content about their products or services. The data could be used to understand consumer behavior through visualizations, text analytics and other insights.
Google is also investing heavily in artificial intelligence for its search engine, making it easier for consumers to find relevant information on websites like Google Maps and Bing Ads, while simultaneously improving the way people see ads using images. “We’re building this into our platform so we can make a big

You:However recently AI returns are failing to keep up with investments

GPT:  in advertising and online marketing. In fact, there have only been 3 years worth of efforts being made for the industry to implement these new capabilities,” said Steve Ballmer, CEO of Google.<|endoftext|>A few days ago I visited my local library, looking at books and magazines from all over the world. There were some very interesting titles – but not many. Then I started thinking about what else I might want to read:
I am hoping that I am going to take a break from reading so much I don't get distracted, but I do want to come back again to read more. At least I hope to have time for the


## Current Performance

The model has learned grammar, spelling, punctuation, and sentence structure well, producing fluent and readable text. It is able to keep track of context for short paragraphs and produces coherent text. Long term context retention is missing topic drift, hallucination is present.
Presently the model has no factual or reasoning abilities which are expected given the training budget.

## Future Work

The next objective is to create custom kernels to fuse operations and train on more data
ideally doubling the size to 400M and training on the full 10B sample. 

