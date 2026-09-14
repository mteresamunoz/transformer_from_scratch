# `feedforward.py` + `block.py` — packaging attention into a repeatable unit

## `feedforward.py`: per-position non-linearity

After attention, every position already holds a mix of relevant information from earlier positions — but that mix is purely **linear** (weighted sums). The feedforward network adds non-linearity and extra capacity to "think" about that information, one position at a time, independently — unlike attention, it never mixes information *between* positions.

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),   # expand
    nn.GELU(),                        # non-linearity
    nn.Linear(4 * n_embd, n_embd),   # compress back down
)
```

Shapes: `(B, T, n_embd) → (B, T, 4*n_embd) → (B, T, n_embd)` — same in/out shape as attention, ready to slot into the residual stream.

### Why `4×` in the hidden layer

The original 2017 paper used `d_ff=2048` with `d_model=512` — a `4×` ratio. This isn't derived from any deep principle — it's what worked well empirically in that paper's experiments, and became the standard convention GPT-2, BERT, and most classic transformers followed without questioning it. It gives the hidden layer more room for non-linear transformations without exploding the parameter count too much — in fact, most of a transformer's parameters live in these feedforward layers, not in attention, precisely because of this `4×` expansion. Modern variants (like SwiGLU, see `GPT.md`) sometimes use a different ratio (e.g. `8/3×`) to compensate for the extra "gate" projection they add.

### GELU vs ReLU

`ReLU(x) = max(0, x)` — an all-or-nothing switch: negative → exactly 0 (zero gradient there, "dead"), positive → passed through unchanged. A hard "elbow" at `x=0`.

**GELU** is a smooth version of that idea: instead of a sharp cutoff, it *weights* the input by "how likely it is worth keeping", using the standard Gaussian's cumulative distribution function. Intuitively:
- Very positive `x` → behaves almost like ReLU (passes nearly everything through).
- Very negative `x` → nearly 0, same as ReLU.
- **Near 0** → a smooth transition (no hard elbow), and it can even output slightly *negative* values for slightly negative inputs — something ReLU never does (ReLU is exactly 0 for any negative input).

Analogy: ReLU is a light switch (on/off); GELU is a dimmer that transitions smoothly, with no abrupt jump. Empirically, this smoothness trains better for language models, which is why GPT-2 and BERT chose it over the original paper's ReLU.

## `block.py`: where "Add & Norm" actually lives

**This is architecture, not training.** `LayerNorm` is a neural network layer — it has its own learnable parameters, lives inside the model, and runs every single time `forward()` is called, whether that call happens during training or during generation. It has nothing to do with `train.py`, which never looks inside a model's `forward()` at all.

"Add & Norm" is two ingredients wrapped *around* each sub-layer (attention, feedforward):

- **Add** = the residual connection: `x = x + sublayer(x)`.
- **Norm** = `LayerNorm`, applied **before** each sub-layer in GPT-2/nanoGPT (*pre-norm* — see the comparison table in `GPT.md`, this differs from the original 2017 paper's *post-norm*).

```python
def forward(self, x):
    x = x + self.attn(self.ln1(x))    # Add & Norm around attention
    x = x + self.ffwd(self.ln2(x))    # Add & Norm around feedforward
    return x
```

Both `ln1` and `ln2` are separate `nn.LayerNorm(n_embd)` instances — each sub-layer gets its own, with its own learnable scale/shift parameters. Shape in, shape out: still `(B, T, n_embd)`, so this can be stacked `n_layer` times with no shape bookkeeping needed.

### Why the residual connection (`x + ...`) matters

Without `+ x`, stacking many blocks would mean the original signal has to survive being transformed correctly by every single block in a row — if any one block's output is a bit off, the error compounds through the whole stack, and gradients have to flow back through every transformation to reach early blocks (which tends to vanish in deep networks). With `x + sublayer(x)`, each block only has to learn a *correction* to add on top of what's already there, and gradients have a direct path (`+`) straight back to earlier layers during `backward()`, bypassing the sublayer's transformation entirely if needed. This is what makes it practical to stack many blocks deep.
