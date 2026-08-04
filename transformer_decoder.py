import torch
import torch.nn as nn
import tiktoken
from datasets import load_dataset
import math
from collections import Counter
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import os


class Transformer(nn.Module):

    def __init__(  # Total parameters: ~95M
        self,
        d_emb=768,
        vocab=50257,
        h=8,
        max_seq_len=512,
        hidden_size=2048,  # 8/3d for swiglu
        batch_size=10,
        n_blocks=8,
    ):
        # Initialize parent class
        super().__init__()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        # self.tokenizer = Tokenisor() # python implentation is too slow
        self.embedding = Embedding(d_emb, vocab)
        self.norm = RMSNorm(d_emb)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_emb, h, max_seq_len, hidden_size, n_blocks)
                for block in range(n_blocks)
            ]
        )
        self.language_modelling = LanguageModelling(self.embedding.E_master)
        self.batch_generator = BatchGenerator(max_seq_len, batch_size)
        self.optimiser = None  # AdamW(self.parameters())

        self.d_emb = d_emb
        self.vocab = vocab
        self.h = h
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.batch_size = batch_size

    def forward(self, text, epochs=25, token_limit=256):
        if self.training:
            self.fit(text, epochs)
        else:
            return self.decoder(text, token_limit)

    def generator(self):
        print("Starting conversation: ")
        history = ""
        while True:
            user_input = input("\nYou:")
            history += user_input
            if user_input.lower() in ["quit", "exit"]:
                break
            response = self.decoder(history)
            print(f"\nGPT: {response}")
            history += response

    @torch.no_grad()
    def decoder(self, text, token_limit=256):

        self.clear_kv_cache()

        token_count = 0

        tokens = torch.tensor(
            self.tokenizer.encode(text), dtype=torch.long, device=device
        ).unsqueeze(
            0
        )  # Shape->(1,n)
        tokens = tokens[:, -self.max_seq_len :]

        # Process prompt
        prompt_len = tokens.shape[1]
        x_emb = self.embedding(tokens)  # Shape->(1,n,d_emb)
        for transformer_block in self.transformer_blocks:
            x_emb = transformer_block(
                x_emb, use_cache=True, pos=0
            )  # Build KV Cache Shape->(1,n,d_emb)
        x_norm = self.norm(x_emb)  # Shape->(1,n,d_emb)

        next_token = self.language_modelling(tokens, x_norm)

        generated = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=device,
        )

        tokens = torch.cat((tokens, generated), dim=1)

        current_token = generated  # scalar int
        pos = tokens.shape[1] - 1
        # Generation
        while token_count < token_limit and pos < (self.max_seq_len - 1):

            x_emb = self.embedding(current_token)  # Shape->(1,n,d_emb)
            for transformer_block in self.transformer_blocks:
                x_emb = transformer_block(x_emb, use_cache=True, pos=pos)
            x_norm = self.norm(x_emb)  # Shape->(1,n,d_emb)
            next_token = self.language_modelling(tokens, x_norm)  # int token id
            tokens = torch.cat(
                [tokens, torch.tensor([[next_token]], device=tokens.device)], dim=1
            )  # Shape->(1,n)
            current_token = torch.tensor(
                [[next_token]], dtype=torch.long, device=device
            )
            pos += 1
            token_count += 1
        generation = tokens[:, prompt_len:]
        response = self.tokenizer.decode(generation.flatten().tolist())
        return response

    def fit(self, tokens, epochs=100, steps=2200, a_steps=10, peak_lr=1e-4):

        # self.tokenizer.byte_pair_encoding(text, self.vocab)

        self.to(device)  # move to gpu

        if self.optimiser is None:
            self.optimiser = AdamW(self.parameters())
        total_steps = epochs * steps
        total_optm_steps = total_steps // a_steps
        warmup = int(0.025 * total_optm_steps)
        start_lr = 0.1 * peak_lr
        min_lr = 0.01 * peak_lr
        step = 0
        optm_step = 0
        print("Training:")
        running_mean_loss = torch.zeros((), device=device)
        for epoch in range(epochs):
            for i in range(steps):
                # Shape -> (B,n)
                batch = self.batch_generator(tokens)

                # move to gpu
                batch = batch.to(device)

                x_emb = self.embedding(batch, step)  # Shape->(B,n,d_emb)
                for transformer_block in self.transformer_blocks:
                    x_emb = transformer_block(x_emb, step)  # Shape->(B,n,d_emb)
                x_norm = self.norm(x_emb)  # Shape->(B,n,d_emb)

                loss = self.language_modelling(batch, x_norm, step)  # compute loss
                scaled_loss = loss / a_steps  # Accumulate and scale
                scaled_loss.backward()  # compute gradients

                running_mean_loss += loss.detach() / a_steps

                if (step + 1) % a_steps == 0 or (step + 1) == total_steps:
                    if optm_step < warmup:  # Warm up
                        lr = start_lr + (optm_step / warmup) * (peak_lr - start_lr)
                    else:  # Cosine lr decay
                        lr = min_lr + 0.5 * (peak_lr - min_lr) * (
                            1
                            + math.cos(
                                math.pi
                                * ((optm_step - warmup) / (total_optm_steps - warmup))
                            )
                        )

                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Clip

                    self.optimiser.update(lr)  # update parameters
                    self.optimiser.clear_grad()  # clear gradients
                    mean_ppl = math.exp(running_mean_loss)
                    if (optm_step) % 10 == 0:
                        print(
                            f"Batch {optm_step} | Loss: {running_mean_loss.item():.4f} | Perplexity: {mean_ppl:.4f}"
                        )
                    running_mean_loss = 0
                    optm_step += 1
                step += 1
            if (epoch) % 5 == 0 and epoch > 0:
                self.save()

            print(
                f"Epoch {epoch} | Loss: {loss.item():.4f} | Perplexity: {torch.exp(loss).item():.4f}"
            )
        self.save()

    def clear_kv_cache(self):
        for block in self.transformer_blocks:
            block.multi_head_attention.k_cache = None
            block.multi_head_attention.v_cache = None

    def save(
        self,
        filename=r"transformer_params.pt",
    ):
        path = Path(__file__).parent / filename
        torch.save(self.state_dict(), path)
        print("Succesfully saved parameters to path")

    def load(
        self,
        filename=r"transformer_params.pt",
    ):
        path = Path(__file__).parent / filename
        self.load_state_dict(torch.load(path, map_location=device))


class TransformerBlock(nn.Module):

    def __init__(self, d_emb, h, max_seq_len, hidden_size, n_blocks):
        # Initialize parent class
        super().__init__()
        self.norm1 = RMSNorm(d_emb)
        self.norm2 = RMSNorm(d_emb)
        self.multi_head_attention = MultiHeadAttention(h, d_emb, max_seq_len, n_blocks)
        self.residual = Residual()
        self.ffn = FFN(d_emb, hidden_size, n_blocks)

    def forward(self, X, step, use_cache=False, pos=0):
        x_norm1 = self.norm1(X)  # Shape -> (B,n,d_emb)
        a = self.multi_head_attention(
            x_norm1, step, use_cache, pos
        )  # Shape -> (B,n,d_emb)
        z = self.residual(X, a)  # Shape -> (B,n,d_emb)
        z_norm = self.norm2(z)  # Shape -> (B,n,d_emb)
        f = self.ffn(z_norm, step)  # Shape -> (B,n,d_emb)
        y = z + f

        return y


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_emb, max_seq_len, n_blocks):
        # Initialize parent class
        super().__init__()

        self.h = h
        self.d_emb = d_emb
        self.d_h = d_emb // h

        self.scale_emb = 1 / math.sqrt(d_emb)
        self.scale_kq = 1 / math.sqrt(self.d_h)
        self.scale_residual = 1 / math.sqrt(2 * n_blocks)

        self.register_buffer(  # Buffer -> Not a parameter, moves devices automatically, caches
            "mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )
        self.positional_encoding = PositionalEncoding(max_seq_len, d_emb // h)

        self.dropout = Dropout()

        self.W_qkv_master = nn.Parameter(torch.randn(d_emb, 3 * d_emb) * self.scale_emb)
        self.W_o_master = nn.Parameter(
            torch.randn(d_emb, d_emb) * self.scale_emb * self.scale_residual
        )

        self.k_cache = None
        self.v_cache = None

    def forward(self, X, step=0, use_cache=False, pos=0):
        # bf16 weights
        if step % 10 == 0 and self.training:
            self.W_qkv = self.W_qkv_master.to(torch.bfloat16)
            self.W_o = self.W_o_master.to(torch.bfloat16)
        if not self.training:
            self.W_qkv = self.W_qkv_master.to(torch.bfloat16)
            self.W_o = self.W_o_master.to(torch.bfloat16)

        QKV = X @ self.W_qkv  # Shape->(B,n,3*d_emb)
        Q, K, V = torch.chunk(QKV, 3, dim=-1)  # Shape->(B,n,d_emb)

        B, n, d_emb = X.shape
        h, d = self.h, self.d_h

        Q = Q.reshape(B, n, h, d).transpose(1, 2)  # Shape->(B,h,n,d_h)
        K = K.reshape(B, n, h, d).transpose(1, 2)  # Shape->(B,h,n,d_h)
        V = V.reshape(B, n, h, d).transpose(1, 2)  # Shape->(B,h,n,d_h)

        Q, K = self.positional_encoding(Q, K, pos=pos)

        if use_cache:
            if self.k_cache is not None:
                self.k_cache = torch.cat((self.k_cache, K), dim=2)
                self.v_cache = torch.cat((self.v_cache, V), dim=2)
            else:
                self.k_cache = K.detach()
                self.v_cache = V.detach()

            K_ = self.k_cache
            V_ = self.v_cache
        else:
            K_ = K
            V_ = V
        # Attention Score
        Z = (Q @ K_.transpose(-2, -1)) * self.scale_kq  # Shape->(B,h,n,n)
        # Decoder Mask
        if n > 1:
            mask = self.mask[:n, :n]
            Z = Z.masked_fill(mask, float("-inf"))

        # Softmax
        Z = Z.to(torch.float32)  # upscale
        S = torch.softmax(Z, dim=-1)  # Shape->(B,h,n,n)
        if self.training:
            S = self.dropout(S)  # Attention dropout
        S = S.to(torch.bfloat16)  # downscale

        A = S @ V_  # Shape->(B,h,n,d_h)
        A = A.transpose(1, 2).reshape(B, n, d_emb)  # Shape->(B,n,d_emb)

        y = A @ self.W_o  # Shape->(B, n, d_emb)
        if self.training:
            y = self.dropout(y)  # Output dropout

        return y


class AdamW:
    def __init__(self, parameters, beta1=0.9, beta2=0.95, w_decay=0.01):
        self.params = list(parameters)
        self.beta1 = beta1
        self.beta2 = beta2
        self.w_decay = w_decay
        self.momentum = [torch.zeros_like(p) for p in self.params]
        self.velocity = [torch.zeros_like(p) for p in self.params]
        self.t = 1
        self.e = 1e-8

    def clear_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None

    def update(self, lr=2e-4):
        with torch.no_grad():  # disable computational graph
            # bias correction
            bc1 = 1 - self.beta1**self.t
            bc2 = 1 - self.beta2**self.t
            for i, p in enumerate(self.params):
                g = p.grad
                if g is None:
                    continue
                if g.dtype == torch.bfloat16:
                    g = g.to(torch.float32)

                # Weight Decay
                if self.w_decay != 0:
                    p.add_(p, alpha=-lr * self.w_decay)

                # Compute 1st and 2nd moment
                self.momentum[i].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                self.velocity[i].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)

                # Bias correction
                m_hat = self.momentum[i] / bc1
                v_hat = self.velocity[i] / bc2

                # Update
                p.addcdiv_(m_hat, v_hat.sqrt_().add_(self.e), value=-lr)

            self.t += 1


class Tokenizer:  # Byte Pair Encoding
    def __init__(self):
        self.merge_rank = {}  # pair -> merge order
        self.merge_to_id = {}  # pair -> token id
        self.id_to_merge = {}  # token id -> pair
        self.merges = []

    def byte_pair_encoding(self, texts, vocab_size):
        print("Byte Pair Encoding started")
        corpus = [list(text.encode("utf-8")) for text in texts]

        next_token = 256

        while next_token < vocab_size:
            pair_counts = self.count_pairs(corpus)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)

            self.merge_to_id[best_pair] = next_token
            self.id_to_merge[next_token] = best_pair
            self.merge_rank[best_pair] = len(self.merges)
            self.merges.append(best_pair)

            corpus = self.merge_corpus(corpus, best_pair)

            next_token += 1

    def encode(self, text):  # text to tokens
        tokens = list(text.encode("utf-8"))

        while True:
            best_pair = None
            best_rank = float("inf")

            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_rank.get(pair)

                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            tokens = self.merge_sequence(tokens, best_pair)

        return torch.tensor(tokens, dtype=torch.long)

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def decode(self, ids):  # token ids to text
        decoded = []
        for token in ids:
            decoded.extend(self.expand_token(int(token)))
        return bytes(decoded).decode("utf-8", errors="replace")

    @lru_cache(maxsize=None)
    def expand_token(self, token):
        if token < 256:
            return [token]
        else:
            left, right = self.id_to_merge[token]
        return self.expand_token(left) + self.expand_token(right)

    def count_pairs(self, corpus):
        counts = Counter()  # accounts for missing keys
        for seq in corpus:
            for pair in zip(seq, seq[1:]):
                counts[pair] += 1
        return counts

    def merge_corpus(self, corpus, pair):
        return [self.merge_sequence(seq, pair) for seq in corpus]

    def merge_sequence(self, seq, pair):
        i = 0
        merged = []
        new_token = self.merge_to_id[pair]

        while i < len(seq):
            if i < (len(seq) - 1) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(seq[i])
                i += 1
        return merged

    def save(
        self,
        filename=r"tokenisor.json",
    ):
        path = Path(__file__).parent / filename
        data = {"merges": [[a, b] for (a, b) in self.merges]}

        with open(path, "w") as f:
            json.dump(data, f)

    def load(
        self,
        filename=r"tokenisor.json",
    ):
        path = Path(__file__).parent / filename
        with open(path, "r") as f:
            data = json.load(f)

        self.merges = [tuple(pair) for pair in data["merges"]]

        next_token = 256

        for rank, pair in enumerate(self.merges):
            self.merge_rank[pair] = rank
            self.merge_to_id[pair] = next_token
            self.id_to_merge[next_token] = pair

            next_token += 1


class BatchGenerator(nn.Module):
    def __init__(self, max_seq_len, batch_size):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.offsets = np.arange(max_seq_len)

    def forward(self, tokens):  # Shape -> (N,)
        N = len(tokens)
        starts = np.random.randint(0, N - self.max_seq_len - 1, size=self.batch_size)
        index = starts[:, None] + self.offsets
        batch = torch.as_tensor(tokens[index], dtype=torch.long)
        return batch  # Shape -> (batch_size,n)


class Embedding(nn.Module):
    def __init__(self, d_emb, vocab):
        # Initialize parent class
        super().__init__()

        self.E_master = nn.Parameter(
            torch.randn((vocab, d_emb)) * (1 / math.sqrt(d_emb))
        )

    def forward(self, X, step=0):
        # bf16 weights
        if step % 10 == 0 and self.training:
            self.E = self.E_master.to(torch.bfloat16)
        if not self.training:
            self.E = self.E_master.to(torch.bfloat16)
        X = X.to(self.E.device)
        return self.E[X]


class PositionalEncoding(nn.Module):  # ROPE
    def __init__(self, max_seq_len, d_h):
        # Initialize parent class
        super().__init__()

        m = torch.arange(max_seq_len, dtype=torch.bfloat16)
        theta = 10000 ** ((-2 * torch.arange(0, d_h, 2, dtype=torch.bfloat16)) / d_h)

        angles = torch.outer(m, theta)
        angles = torch.cat((angles, angles), dim=-1)

        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, Q, K, pos=0):
        n = Q.shape[-2]
        d = Q.shape[-1]

        cos = self.cos[pos : pos + n, :d].unsqueeze(0)
        sin = self.sin[pos : pos + n, :d].unsqueeze(0)

        assert d % 2 == 0

        Q1, Q2 = Q.chunk(2, dim=-1)
        Q_r = torch.cat((-Q2, Q1), dim=-1)

        K1, K2 = K.chunk(2, dim=-1)
        K_r = torch.cat((-K2, K1), dim=-1)

        Q = Q * cos + Q_r * sin
        K = K * cos + K_r * sin

        return Q, K


class RMSNorm(nn.Module):
    def __init__(self, d_emb):
        # Initialize parent class
        super().__init__()

        # Learnable Parameter
        self.gamma = nn.Parameter(torch.ones(1, 1, d_emb))

    def forward(self, X):  # Normalize using rms
        if X.dtype == torch.bfloat16:
            X = X.to(torch.float32)
        B, N, d_emb = X.shape
        rms = self.rms(X)
        norm_x = X / rms * self.gamma
        return norm_x.to(torch.bfloat16)

    def rms(self, x):
        B, N, d_emb = x.shape
        mean = (1 / d_emb) * torch.sum(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean + 1e-9)
        return rms


class Dropout(nn.Module):
    def __init__(self, keep=0.9):
        # Initialize parent class
        super().__init__()
        self.keep = keep

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) < self.keep).to(x.dtype)
            x_drop = (x * mask) / self.keep
            return x_drop
        else:
            return x


class Residual(nn.Module):
    def __init__(self):
        # Initialize parent class
        super().__init__()

    def forward(self, x, a):
        return x + a


class FFN(nn.Module):  # Swiglu

    def __init__(self, d_emb, hidden_size, n_blocks):
        # Initialize parent class
        super().__init__()

        self.scale_residual = 1 / math.sqrt(2 * n_blocks)

        self.W1_master = nn.Parameter(
            torch.randn(d_emb, hidden_size) * (1 / math.sqrt(hidden_size))
        )
        self.W2_master = nn.Parameter(
            torch.randn(d_emb, hidden_size) * (1 / math.sqrt(hidden_size))
        )
        self.W3_master = nn.Parameter(
            torch.randn(hidden_size, d_emb)
            * (1 / math.sqrt(hidden_size) * self.scale_residual)
        )

        self.silu = nn.SiLU()

        self.dropout = Dropout()

    def forward(self, X, step=0):
        # bf16 weights
        if step % 10 == 0 and self.training:
            self.W1 = self.W1_master.to(torch.bfloat16)
            self.W2 = self.W2_master.to(torch.bfloat16)
            self.W3 = self.W3_master.to(torch.bfloat16)
        if not self.training:
            self.W1 = self.W1_master.to(torch.bfloat16)
            self.W2 = self.W2_master.to(torch.bfloat16)
            self.W3 = self.W3_master.to(torch.bfloat16)

        gate = X @ self.W1  # Shape -> (B,n,hidden)
        up = X @ self.W2  # Shape -> (B,n,hidden)
        h = self.silu(gate) * up  # Shape -> (B,n,hidden)
        down = h @ self.W3  # Shape -> (B,n,d emb)
        if self.training:
            down = self.dropout(down)
        return down


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        # Initialize parent class
        super().__init__()

    def forward(self, X, Y):
        B, n = X.shape  # X Shape->(B, n) -> token ids for text batches
        logits = Y[:, :-1, :]  # Softmaxed logits Shape->(B, n-1)
        targets = X[:, 1:]  # Sliced token ids Shape->(B, n-1)

        log_probs = torch.log_softmax(logits, dim=-1)

        # for b in range(B): # Naive Implementation
        #     for t in range(n):
        #         loss += -torch.log(probs[b, t, targets[b, t]])
        # loss /= (B * n)

        selected = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        loss = -selected.mean()
        return loss


class LanguageModelling(nn.Module):
    def __init__(self, emb):
        # Initialize parent class
        super().__init__()
        self.cross_entropy_loss = CrossEntropyLoss()
        self.emb_master = emb  # Shape->(vocab,d_emb)
        self.loss = 0

    def forward(self, token_ids, y, step=0, temp=0.6):  # y->(B, n, emb)
        if temp <= 0:  # Temperature Scaling
            temp = 1  # Standard softmax

        if self.training:
            if step % 10 == 0:
                self.emb = self.emb_master.to(torch.bfloat16)  # Shape->(vocab,d_emb)
            temp = 1  # Standard softmax
            logits = y @ self.emb.T  # Shape->(B, n, vocab)
            if logits.dtype == torch.bfloat16:
                logits = logits.to(torch.float32)
            # Shape->(B, n, vocab)
            # S = torch.exp(logits / temp) / torch.sum(logits / temp, dim=-1, keepdim=True)  # prob

            self.loss = self.cross_entropy_loss(
                token_ids, logits
            )  # Token id's Shape->(B, n)
            return self.loss
        else:  # generation
            self.emb = self.emb_master.to(torch.bfloat16)  # Shape->(vocab,d_emb)
            # y shape-> (1,n,d_emb)
            y_last = y[:, -1, :]
            logit = y_last @ self.emb.T
            if logit.dtype == torch.bfloat16:
                logit = logit.to(torch.float32)
            k = 50  # Top K Sampling
            topk, _ = torch.topk(logit, k)
            threshold = topk[:, [-1]]
            logit[logit < threshold] = float("-inf")
            probs = torch.softmax(logit / temp, dim=-1)  # Shape->(B, n, vocab)
            next_token = torch.multinomial(probs, num_samples=1).item()
            return next_token  # integer token id


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(target=1200000000, filepath="fineweb-tokenised.bin"):
    if os.path.exists(filepath):
        print("Found tokenised file")
        tokens_memmap = np.memmap(filepath, dtype=np.uint16, mode="r")
        print("Loaded tokenised file")
        return tokens_memmap
    print("tokenising file")
    tokenisor = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )

    path = Path(__file__).parent / filepath

    total_tokens = 0
    with open(path, "wb") as f:
        for row in dataset:
            tokens = tokenisor.encode_ordinary(row["text"])
            tokens.append(50256)

            tokens_np = np.array(tokens, dtype=np.uint16)
            f.write(tokens_np.tobytes())

            total_tokens += len(tokens)

            if total_tokens >= target:
                break
    print(f"Finished tokenizing {total_tokens:,} tokens and saved to {path}")

    return np.memmap(filepath, dtype=np.uint16, mode="r")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


tokens = load_data()


model = Transformer()
total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params:,}")

# model.load()

model = torch.compile(model, mode="max-autotune")

model.train()
model.fit(tokens)


# model.eval()
# model.generator()
