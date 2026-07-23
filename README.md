# Transformer From Scratch

## Overview

This project is a decoder-only Transformer implemented almost entirely from scratch to better understand how modern language models work at a low level.

PyTorch is only used for automatic differentiation, tensor operations, and GPU acceleration. Core components including multi-head self-attention, RoPE positional encoding, RMS Norm, dropout, Swiglu, AdamW, cross-entropy loss, KV caching, batching, and the training loop—are implemented manually. The current model contains approximately **30 million trainable parameters** and supports both training and autoregressive text generation.

## Learning Outcomes

Building the model highlighted that implementation details are often more challenging than the underlying mathematics. I encountered bugs involving device management, where my custom AdamW optimizer remained on the CPU after the model was moved to the GPU, as well as several RoPE implementation bugs caused by incorrect tensor reshaping and inference positional indexing, all of which prevented the model from training or generating correctly until resolved.

As the project matured, I focused on optimization as well as correctness. I replaced separate key, query, and value projections for each attention head with a single fused QKV projection followed by reshaping into a 4D tensor, making the implementation both cleaner and more efficient. Implementing AdamW from scratch provided a much deeper understanding of momentum, variance estimation, bias correction, and weight decay than simply using the PyTorch implementation.

I also experimented with building my own Byte Pair Encoding tokenizer. While it functioned correctly, the pure Python implementation was far too slow for practical training, so I switched to **tiktoken** while keeping the remainder of the model implemented from scratch.

Training required several optimization improvements. A fixed learning rate consistently stalled around a loss of **3.8**, so I introduced warmup and cosine learning-rate decay, allowing training to converge to approximately **3.5**. GPU memory limitations on my RTX 4060 prevented using larger batches, so I implemented gradient accumulation to achieve larger effective batch sizes while reducing optimization volatility. Update(Switched layer norm and ffn to rms norm and swiglu)

## Training Progress

The model was initially trained on WikiText-103 to validate the implementation. The loss decreased from approximately **10** to **6** early in training, continued through **5 → 4**, plateaued briefly around **4**, and eventually converged to roughly **3.5**.

## Features

* Decoder-only Transformer
* Multi-head self-attention
* Rotary Positional Embeddings (RoPE)
* KV cache for efficient autoregressive inference
* Pre-LayerNorm architecture
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
PS C:\Users\neelb\Documents\CS> & C:/Users/neelb/AppData/Local/Programs/Python/Python313/python.exe c:/Users/neelb/Documents/CS/Projects/Transformer/transformer_decoder.py
Total trainable parameters: 29,937,024
Starting conversation: 

You:the capital of france is

GPT:  a place of thought , so that no one can be seen in any single person . 


 = = = = Modern = = = = 


 The earliest known modernist novel , The Great Man , is the novel of the same name , in which the person is called the " Old Woman " or " Old Woman " . The first novel , The Great Man , was published in 1794 ; it was published in 1796 . The novel is a play in which the narrator is a woman who is given access to her marriage , and the play was published in 1794 . The novel was written in 1790 , and was published in 1790 . It was published in 1796 , and was translated into English , but by the time it was published , it was published in 1797 . 

 The novel has been adapted from a novel by William Blake and published by the English professor John R. <unk> , who wrote it in 1793 . The novel was published in 1793 and was published in 1794 . The novel was published in 1794 by Samuel <unk> , who was the first publisher of The Great Man . The novel was written by John R. <unk> , the author of The Great Man , who wrote it in 1795 ,

You:america was discovered by

GPT:  an unknown man . R. <unk> , who wrote the novel in 1795 , wrote The Great Man in 1797 . The book was published in 1794 by John R. <unk> , who wrote the book in 1795 . R. <unk> , a writer of The Great Man , wrote the novel in 1797 for the Danish newspaper <unk> , which was published in 1796 . R. <unk> , author of The Great Man , wrote the novel in 1798 . 


 = = = = Modern = = = = 


 Modern scholars have found that the novel is considered one of the most famous English works in the world . They argue that the novel was based on a story about a man who is called " the man who has the right to leave his home . " The novel is not a " book of the same name , " but it is not a poem . 


 = = = Modern period = = = 


 The novel was originally written as a novel by a friend of William Blake , who wrote a letter to his friend , John R. <unk> , and was published in 1796 . The novel was written by John R

You:quit
PS C:\Users\neelb\Documents\CS> 

## Current Performance

The model has learned grammar, spelling, punctuation, and sentence structure well, producing fluent and readable text. Its main limitation is **long-range context retention**, as generation is strongly influenced by recently generated tokens, often causing **topic drift**. The primary bottlenecks are now **training data** and **compute** rather than the architecture itself.

## Future Work

The next objective is to train the model on the **FineWeb** dataset using rented GPUs. Reducing the training loss from **~3.5** toward **3.0–2.5** should significantly improve coherence, context retention, and factual consistency while providing a better understanding of how scaling data and compute affects model performance.
