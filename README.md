# Mathematical Descriptions of All Models

**Authors:** Satabarto Sarkar, Prithwineel Paul, Gexiang Zhang, Ferrante Neri, Antonio Ramírez de Arellano Marrero, Agustín Riscos‑Núñez, David Orellana‑Martín

---

## Table of Contents

1. [LSTM‑SNP Cell (Baseline)](#1-lstm-snp-cell--baseline)
2. [Type 1 — LSTM‑SNP Baseline Model](#2-type-1--lstm-snp-baseline-model)
3. [Type 2 — Fuzzy Feature Augmentation](#3-type-2--fuzzy-feature-augmentation)
4. [Type 3 — Fuzzy Gate Replacement (Clamp‑Bounded)](#4-type-3--fuzzy-gate-replacement-clamp-bounded)
5. [Type 3 Sigmoid — Fuzzy Gate Replacement (Sigmoid‑Bounded)](#5-type-3-sigmoid--fuzzy-gate-replacement-sigmoid-bounded)
6. [Type 4 — Fuzzy Output Layer](#6-type-4--fuzzy-output-layer)
7. [Type 5 — Hybrid Fuzzy Feature Augmentation + Fuzzy Gate Replacement](#7-type-5--hybrid-fuzzy-feature-augmentation--fuzzy-gate-replacement)
8. [Pure LSTM](#8-pure-lstm)
9. [Pure GRU](#9-pure-gru)
10. [Bidirectional LSTM‑SNP (BiLSTM‑SNP)](#10-bidirectional-lstm-snp-bilstm-snp)
11. [LSTM‑SNP (Sequence‑over‑Lag)](#11-lstm-snp-sequence-over-lag)
12. [SNN Transformer (Spike‑Driven Transformer)](#12-snn-transformer-spike-driven-transformer)
13. [Attention (Pure LSTM with Lag Experiments)](#13-attention-pure-lstm-with-lag-experiments)

---

## 1. LSTM‑SNP Cell — Baseline

> [!IMPORTANT]
> The **LSTM‑SNP cell** is the foundational recurrent unit shared across Types 1–5. It is inspired by Spiking Neural P Systems and differs from the standard LSTM by replacing the cell state with a **membrane potential** $\mathbf{u}(t)$ and using an **additive‑subtractive** update rule.

### 1.1 Pre‑activation

Given input $\mathbf{x}(t) \in \mathbb{R}^{d_{\text{in}}}$ and previous membrane potential $\mathbf{u}(t{-}1) \in \mathbb{R}^{H}$ (where $H$ is the hidden size), compute the combined pre‑activation:

$$\mathbf{z}(t) = \mathbf{W}\,\mathbf{x}(t) + \mathbf{U}\,\mathbf{u}(t{-}1) + \mathbf{b}$$

where $\mathbf{W} \in \mathbb{R}^{4H \times d_{\text{in}}}$, $\mathbf{U} \in \mathbb{R}^{4H \times H}$, $\mathbf{b} \in \mathbb{R}^{4H}$.

$\mathbf{W}$ is initialized with Xavier uniform; $\mathbf{U}$ is initialized with orthogonal initialization.

### 1.2 Gate Decomposition

Split $\mathbf{z}(t)$ into four $H$‑dimensional slices:

$$\mathbf{z}_0,\; \mathbf{z}_1,\; \mathbf{z}_2,\; \mathbf{z}_3 = \text{chunk}(\mathbf{z}(t),\; 4)$$

### 1.3 Gate Activations

| Gate | Symbol | Activation | Role |
|------|--------|-----------|------|
| **Reset** | $\mathbf{r}(t)$ | $\text{hardsigmoid}(\mathbf{z}_0)$ | Controls retention of previous membrane potential |
| **Consumption** | $\mathbf{c}(t)$ | $\text{hardsigmoid}(\mathbf{z}_1)$ | Controls absorption of generated spikes |
| **Output/Generation** | $\mathbf{o}(t)$ | $\text{hardsigmoid}(\mathbf{z}_2)$ | Controls spike emission |
| **Generated Spikes** | $\mathbf{a}(t)$ | $\tanh(\mathbf{z}_3)$ | Spike generation (analogous to candidate cell state) |

where the **hard sigmoid** is defined as:

$$\text{hardsigmoid}(x) = \text{clip}(0.2x + 0.5,\; 0,\; 1)$$

### 1.4 State Update (Membrane Potential)

$$\mathbf{u}(t) = \mathbf{r}(t) \odot \mathbf{u}(t{-}1) \;-\; \mathbf{c}(t) \odot \mathbf{a}(t)$$

> [!NOTE]
> Unlike the standard LSTM cell state update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ (additive), the LSTM‑SNP uses a **subtractive** mechanism ($-$), modeling the biological process where a neuron's membrane potential *decreases* when it consumes (fires) spikes.

### 1.5 Output (Emitted Spikes)

$$\mathbf{h}(t) = \mathbf{o}(t) \odot \mathbf{a}(t)$$

### 1.6 Bias Initialization

The consumption gate bias $\mathbf{b}_{[H:2H]}$ is initialized to $1.0$ (analogous to the forget gate bias initialization in standard LSTMs), ensuring the cell initially retains information.

---

## 2. Type 1 — LSTM‑SNP Baseline Model

**Folder:** [`type_1_baseline_test_60/`](file:///tmp/Fuzzy-LSTM_SNP-Test_LAST60/type_1_baseline_test_60)

### Architecture

$$\hat{y}(t) = \mathbf{w}_{\text{out}}^{\top}\,\mathbf{h}(t) + b_{\text{out}}$$

where $\mathbf{h}(t)$ is the output of the unmodified LSTM‑SNP cell (Section 1).

| Component | Specification |
|-----------|--------------|
| Cell | `LSTMSNPCell` (unmodified) |
| Input dimension | $d_{\text{in}} = 1$ (univariate, first‑order differenced, MinMax‑scaled) |
| Hidden units | $H = 8$ |
| Output layer | `Linear(8, 1)` |
| Stateful | Yes — $\mathbf{u}(t)$ persists across time steps |

### Training Protocol

- **Optimizer:** Adam ($\eta = 0.001$)
- **Loss:** MSE
- **Epochs:** 100 (with early stopping, patience = 10)
- **Gradient clipping:** $\|\nabla\|_{\max} = 1.0$
- **Runs:** 60 independent runs (different random seeds)
- **BPTT truncation:** State detached after each sample (truncated BPTT with window = 1)

---

## 3. Type 2 — Fuzzy Feature Augmentation

**Folder:** [`TYPE_2_TEST_60_NOISE/`](file:///tmp/Fuzzy-LSTM_SNP-Test_LAST60/TYPE_2_TEST_60_NOISE)

> [!TIP]
> In this variant, the LSTM‑SNP cell is **completely unmodified**. Instead, the *input* is augmented with a fuzzy‑inferred feature computed during preprocessing.

### 3.1 Fuzzy Inference System (Preprocessing)

#### Membership Functions (Fixed Gaussian)

$$\mu_{\text{low}}(x) = \exp\!\Bigl(-\frac{(x - (-1))^2}{2 \cdot 0.5^2}\Bigr), \qquad \mu_{\text{high}}(x) = \exp\!\Bigl(-\frac{(x - (+1))^2}{2 \cdot 0.5^2}\Bigr)$$

Centers: $c_{\text{low}} = -1.0$, $c_{\text{high}} = +1.0$. Width: $\sigma = 0.5$. These are **not learned**.

#### Takagi–Sugeno Rules (4 rules, fixed consequents)

Given current input $x(t)$ and previous input $x(t{-}1)$ (both scaled):

| Rule | Antecedent | Consequent |
|------|-----------|------------|
| $R_1$ | $x(t)$ is Low **AND** $x(t{-}1)$ is Low | $y_1 = 0.5\,x(t) + 0.5\,x(t{-}1)$ |
| $R_2$ | $x(t)$ is Low **AND** $x(t{-}1)$ is High | $y_2 = 0.7\,x(t) + 0.3\,x(t{-}1) - 0.1$ |
| $R_3$ | $x(t)$ is High **AND** $x(t{-}1)$ is Low | $y_3 = 0.3\,x(t) + 0.7\,x(t{-}1) + 0.1$ |
| $R_4$ | $x(t)$ is High **AND** $x(t{-}1)$ is High | $y_4 = 0.5\,x(t) + 0.5\,x(t{-}1)$ |

#### Rule Firing Strengths (Product T‑norm)

$$w_1 = \mu_{\text{low}}(x(t)) \cdot \mu_{\text{low}}(x(t{-}1))$$
$$w_2 = \mu_{\text{low}}(x(t)) \cdot \mu_{\text{high}}(x(t{-}1))$$
$$w_3 = \mu_{\text{high}}(x(t)) \cdot \mu_{\text{low}}(x(t{-}1))$$
$$w_4 = \mu_{\text{high}}(x(t)) \cdot \mu_{\text{high}}(x(t{-}1))$$

#### Defuzzification (Weighted Average)

$$y_{\text{fuzzy}}(t) = \frac{\sum_{i=1}^{4} w_i \, y_i}{\sum_{i=1}^{4} w_i + \epsilon}$$

where $\epsilon = 10^{-8}$ for numerical stability.

### 3.2 Augmented Input

The input to the LSTM‑SNP cell becomes 2‑dimensional:

$$\tilde{\mathbf{x}}(t) = \bigl[x(t),\; y_{\text{fuzzy}}(t)\bigr]^{\top} \in \mathbb{R}^2$$

### 3.3 Model Architecture

| Component | Specification |
|-----------|--------------|
| Cell | `LSTMSNPCell` (**unmodified**) |
| Input dimension | $d_{\text{in}} = 2$ (original + fuzzy feature) |
| Hidden units | $H = 8$ |
| Output layer | `Linear(8, 1)` |

---

## 4. Type 3 — Fuzzy Gate Replacement (Clamp‑Bounded)

**Folder:** [`type_3/`](file:///tmp/Fuzzy-LSTM_SNP-Test_LAST60/type_3)

> [!IMPORTANT]
> In this variant, the hard sigmoid activations for gates $\mathbf{r}$, $\mathbf{c}$, $\mathbf{o}$ are **replaced** by a 2‑rule Takagi–Sugeno fuzzy inference system with **trainable** consequent parameters. Gate $\mathbf{a}$ retains $\tanh$.

### 4.1 Gate‑Level Fuzzy Inference

For each gate $g \in \{r, c, o\}$, given the pre‑activation slice $\mathbf{z}_g \in \mathbb{R}^H$ and the mean membrane potential $\bar{u} = \text{mean}(\mathbf{u}(t{-}1)) \cdot \mathbf{1}_H$:

#### Gaussian Membership Functions (Fixed)

$$\mu_{\text{low}}(\mathbf{z}_g) = \exp\!\Bigl(-\frac{(\mathbf{z}_g - (-1))^2}{2\sigma^2}\Bigr), \qquad \mu_{\text{high}}(\mathbf{z}_g) = \exp\!\Bigl(-\frac{(\mathbf{z}_g - (+1))^2}{2\sigma^2}\Bigr)$$

where $\sigma \in \{0.25, 0.5, 0.75, 1.0\}$ is a hyperparameter (experiments sweep over all values).

#### Takagi–Sugeno Rules (2 rules, trainable consequents)

$$y_0^{(g)} = \mathbf{a}_0^{(g)} \odot \mathbf{z}_g + \mathbf{b}_0^{(g)} \odot \bar{u} + \mathbf{c}_0^{(g)}$$
$$y_1^{(g)} = \mathbf{a}_1^{(g)} \odot \mathbf{z}_g + \mathbf{b}_1^{(g)} \odot \bar{u} + \mathbf{c}_1^{(g)}$$

where $\mathbf{a}_k^{(g)}, \mathbf{b}_k^{(g)} \in \mathbb{R}^H$ are initialized $\sim \mathcal{U}(-0.1, 0.1)$ and $\mathbf{c}_k^{(g)} \in \mathbb{R}^H$ is initialized to $\mathbf{0}$.

#### Defuzzification + Bounding

$$g_{\text{raw}} = \frac{\mu_{\text{low}} \odot y_0^{(g)} + \mu_{\text{high}} \odot y_1^{(g)}}{\mu_{\text{low}} + \mu_{\text{high}} + \epsilon}$$

$$g(t) = \text{clip}(g_{\text{raw}},\; 0,\; 1)$$

> [!NOTE]
> The **clamp bounding** $\text{clip}(\cdot, 0, 1)$ ensures gate values remain in $[0, 1]$ like the original hard sigmoid, but with **zero gradient** in the saturated regions. This is the distinguishing feature from the Type 3 Sigmoid variant.

### 4.2 Trainable Parameters per Gate

Each gate $g \in \{r, c, o\}$ has $2$ rules $\times$ $3$ parameter vectors $\times$ $H$ dimensions $= 6H$ trainable consequent parameters. Total fuzzy parameters: $3 \times 6H = 18H$.

### 4.3 Full Cell Equations

$$\mathbf{z}(t) = \mathbf{W}\,\mathbf{x}(t) + \mathbf{U}\,\mathbf{u}(t{-}1) + \mathbf{b}$$

$$\mathbf{r}(t) = \text{FuzzyGate}_r(\mathbf{z}_0,\, \bar{u})$$
$$\mathbf{c}(t) = \text{FuzzyGate}_c(\mathbf{z}_1,\, \bar{u})$$
$$\mathbf{o}(t) = \text{FuzzyGate}_o(\mathbf{z}_2,\, \bar{u})$$
$$\mathbf{a}(t) = \tanh(\mathbf{z}_3)$$

$$\mathbf{u}(t) = \mathbf{r}(t) \odot \mathbf{u}(t{-}1) - \mathbf{c}(t) \odot \mathbf{a}(t)$$
$$\mathbf{h}(t) = \mathbf{o}(t) \odot \mathbf{a}(t)$$

---

## 5. Type 3 Sigmoid — Fuzzy Gate Replacement (Sigmoid‑Bounded)

**Folder:** [`Type 3 Sigmoid/`](file:///tmp/Fuzzy-LSTM_SNP-Test_LAST60/Type%203%20Sigmoid)

This variant is **identical** to Type 3 (Section 4) in every respect except the bounding function:

$$g(t) = \sigma(g_{\text{raw}}) = \frac{1}{1 + e^{-g_{\text{raw}}}}$$

instead of $\text{clip}(g_{\text{raw}}, 0, 1)$.

> [!TIP]
> The **sigmoid bounding** provides smooth, non‑zero gradients everywhere (no dead zones), which can improve gradient flow during training compared to the hard clamp. However, the output is *asymptotically* bounded to $(0, 1)$ rather than *exactly* bounded to $[0, 1]$.

### Key Difference Summary

| Property | Type 3 (Clamp) | Type 3 Sigmoid |
|----------|---------------|----------------|
| Bounding | $\text{clip}(\cdot, 0, 1)$ | $\sigma(\cdot)$ |
| Gradient at saturation | $0$ | $\sigma(x)(1 - \sigma(x)) > 0$ |
| Exact range | $[0, 1]$ | $(0, 1)$ |

---

## 6. Type 4 — Fuzzy Output Layer

**Folder:** [`Type_4/`](file:///tmp/Fuzzy-LSTM_SNP-Test_LAST60/Type_4)

> [!IMPORTANT]
> In this variant, the LSTM‑SNP cell is **unmodified**. Instead, the `Linear(H, 1)` output layer is replaced by a **fuzzy inference system** that maps the $H$‑dimensional hidden output to a scalar prediction.

### 6.1 Summary Features

The hidden output $\mathbf{h}(t) \in \mathbb{R}^H$ is aggregated into two summary statistics via mean‑pooling:

$$s_1 = \frac{1}{H/2}\sum_{j=1}^{H/2} h_j(t), \qquad s_2 = \frac{1}{H/2}\sum_{j=H/2+1}^{H} h_j(t)$$

### 6.2 Membership Functions (Fixed Gaussian)

$$\mu_{\text{low}}(s) = \exp\!\Bigl(-\frac{(s - (-1))^2}{2 \cdot 0.5^2}\Bigr), \qquad \mu_{\text{high}}(s) = \exp\!\Bigl(-\frac{(s - (+1))^2}{2 \cdot 0.5^2}\Bigr)$$

### 6.3 Takagi–Sugeno Rules (4 rules, trainable consequents)

| Rule | Antecedent | Consequent |
|------|-----------|------------|
| $R_1$ | $s_1$ is Low **AND** $s_2$ is Low | $y_1 = a_1 s_1 + b_1 s_2 + c_1$ |
| $R_2$ | $s_1$ is Low **AND** $s_2$ is High | $y_2 = a_2 s_1 + b_2 s_2 + c_2$ |
| $R_3$ | $s_1$ is High **AND** $s_2$ is Low | $y_3 = a_3 s_1 + b_3 s_2 + c_3$ |
| $R_4$ | $s_1$ is High **AND** $s_2$ is High | $y_4 = a_4 s_1 + b_4 s_2 + c_4$ |

where $a_i, b_i \in \mathbb{R}$ are initialized with Glorot uniform ($\pm\sqrt{6/8}$) and $c_i$ are initialized to $0$.

### 6.4 Rule Firing Strengths

$$w_1 = \mu_{\text{low}}(s_1) \cdot \mu_{\text{low}}(s_2)$$
$$w_2 = \mu_{\text{low}}(s_1) \cdot \mu_{\text{high}}(s_2)$$
$$w_3 = \mu_{\text{high}}(s_1) \cdot \mu_{\text{low}}(s_2)$$
$$w_4 = \mu_{\text{high}}(s_1) \cdot \mu_{\text{high}}(s_2)$$

### 6.5 Defuzzification

$$\hat{y}(t) = \frac{\sum_{i=1}^{4} w_i \, y_i}{\sum_{i=1}^{4} w_i + \epsilon}$$

### 6.6 Trainable Parameters

$4$ rules $\times$ $3$ scalar parameters $(a_i, b_i, c_i)$ = **12** trainable parameters in the fuzzy output layer (replacing a single `Linear(8, 1)` with 9 parameters).

---

## 7. Type 5 — Hybrid Fuzzy Feature Augmentation + Fuzzy Gate Replacement

**Folder:** [`Type_5/`](file:///tmp/Fuzzy-LSTM_SNP-Test_LAST60/Type_5)

> [!IMPORTANT]
> Type 5 combines **both** fuzzy enhancements: the fuzzy gate replacement (from Type 3) inside the cell, plus the fuzzy feature augmentation (from Type 2) at the input. This uses the **clamp‑bounded** variant.

### 7.1 Fuzzy Gate Module (`FuzzyGateType5`)

Each gate $g \in \{r, c, o\}$ uses a standalone `FuzzyGateType5` module:

$$\mu_{\text{low}}(\mathbf{z}_g) = \exp\!\Bigl(-\frac{(\mathbf{z}_g + 1)^2}{2\sigma^2}\Bigr), \qquad \mu_{\text{high}}(\mathbf{z}_g) = \exp\!\Bigl(-\frac{(\mathbf{z}_g - 1)^2}{2\sigma^2}\Bigr)$$

$$y_0 = \mathbf{a}_0 \odot \mathbf{z}_g + \mathbf{b}_0 \odot \bar{u} + \mathbf{c}_0$$
$$y_1 = \mathbf{a}_1 \odot \mathbf{z}_g + \mathbf{b}_1 \odot \bar{u} + \mathbf{c}_1$$

$$g(t) = \text{clip}\!\Biggl(\frac{\mu_{\text{low}} \odot y_0 + \mu_{\text{high}} \odot y_1}{\mu_{\text{low}} + \mu_{\text{high}} + \epsilon},\; 0,\; 1\Biggr)$$

with $\sigma = 0.5$ (fixed, not swept).

### 7.2 Cell Equations

Identical to Type 3 (Section 4.3), but with independently parameterized `FuzzyGateType5` modules per gate instead of a shared `ParameterDict`.

### 7.3 Model Architecture

| Component | Specification |
|-----------|--------------|
| Cell | `FuzzyLSTMSNPCell` (fuzzy gates, clamp‑bounded) |
| Input | $d_{\text{in}} = 1$ (despite being "hybrid", the code uses single input in this variant) |
| Hidden units | $H = 8$ |
| Output layer | `Linear(8, 1)` |

---

## 8. Pure LSTM

**Folder:** [`Pure_LSTM/`](file:///tmp/Fuzzy-LSTM_SNP-Test_LAST60/Pure_LSTM)

Standard LSTM (Hochreiter & Schmidhuber, 1997) used as a baseline comparison.

### 8.1 Cell Equations

$$\begin{aligned}
\mathbf{f}(t) &= \sigma\bigl(\mathbf{W}_f \mathbf{x}(t) + \mathbf{U}_f \mathbf{h}(t{-}1) + \mathbf{b}_f\bigr) & \text{(forget gate)} \\
\mathbf{i}(t) &= \sigma\bigl(\mathbf{W}_i \mathbf{x}(t) + \mathbf{U}_i \mathbf{h}(t{-}1) + \mathbf{b}_i\bigr) & \text{(input gate)} \\
\tilde{\mathbf{c}}(t) &= \tanh\bigl(\mathbf{W}_c \mathbf{x}(t) + \mathbf{U}_c \mathbf{h}(t{-}1) + \mathbf{b}_c\bigr) & \text{(candidate)} \\
\mathbf{c}(t) &= \mathbf{f}(t) \odot \mathbf{c}(t{-}1) + \mathbf{i}(t) \odot \tilde{\mathbf{c}}(t) & \text{(cell state)} \\
\mathbf{o}(t) &= \sigma\bigl(\mathbf{W}_o \mathbf{x}(t) + \mathbf{U}_o \mathbf{h}(t{-}1) + \mathbf{b}_o\bigr) & \text{(output gate)} \\
\mathbf{h}(t) &= \mathbf{o}(t) \odot \tanh(\mathbf{c}(t)) & \text{(hidden state)}
\end{aligned}$$

### 8.2 Model

$$\hat{y}(t) = \mathbf{w}_{\text{out}}^{\top}\,\mathbf{h}(t) + b_{\text{out}}$$

| Component | Specification |
|-----------|--------------|
| Cell | `nn.LSTM` (PyTorch built‑in) |
| Input dimension | $d_{\text{in}} = 1$ |
| Hidden units | $H = 8$ |
| Output layer | `Linear(8, 1)` |
| Stateful | Yes — $(h, c)$ persist across time steps |

---

## 9. Pure GRU

**Folder:** [`Pure_GRU/`](file:///tmp/Fuzzy-LSTM_SNP-Test_LAST60/Pure_GRU)

Standard GRU (Cho et al., 2014) used as a baseline comparison.

### 9.1 Cell Equations

$$\begin{aligned}
\mathbf{z}(t) &= \sigma\bigl(\mathbf{W}_z \mathbf{x}(t) + \mathbf{U}_z \mathbf{h}(t{-}1) + \mathbf{b}_z\bigr) & \text{(update gate)} \\
\mathbf{r}(t) &= \sigma\bigl(\mathbf{W}_r \mathbf{x}(t) + \mathbf{U}_r \mathbf{h}(t{-}1) + \mathbf{b}_r\bigr) & \text{(reset gate)} \\
\tilde{\mathbf{h}}(t) &= \tanh\bigl(\mathbf{W}_h \mathbf{x}(t) + \mathbf{U}_h (\mathbf{r}(t) \odot \mathbf{h}(t{-}1)) + \mathbf{b}_h\bigr) & \text{(candidate)} \\
\mathbf{h}(t) &= (1 - \mathbf{z}(t)) \odot \mathbf{h}(t{-}1) + \mathbf{z}(t) \odot \tilde{\mathbf{h}}(t) & \text{(hidden state)}
\end{aligned}$$

### 9.2 Model

$$\hat{y}(t) = \mathbf{w}_{\text{out}}^{\top}\,\mathbf{h}(t) + b_{\text{out}}$$

| Component | Specification |
|-----------|--------------|
| Cell | `nn.GRU` (PyTorch built‑in) |
| Input dimension | $d_{\text{in}} = 1$ |
| Hidden units | $H = 8$ |
| Output layer | `Linear(8, 1)` |
| Stateful | Yes — $\mathbf{h}$ persists across time steps |

---

## 10. Bidirectional LSTM‑SNP (BiLSTM‑SNP)

**Folder:** [`BiLSTM_SNP/`](file:///tmp/Fuzzy-LSTM_SNP-Test_LAST60/BiLSTM_SNP)

### 10.1 Architecture

Two independent LSTM‑SNP cells process the input sequence in opposite temporal directions:

**Forward pass** (left‑to‑right, $t = 1, 2, \ldots, T$):
$$\mathbf{h}_{\text{fw}}(t),\; \mathbf{u}_{\text{fw}}(t) = \text{LSTMSNPCell}_{\text{fw}}\bigl(\mathbf{x}(t),\; \mathbf{u}_{\text{fw}}(t{-}1)\bigr)$$

**Backward pass** (right‑to‑left, $t = T, T{-}1, \ldots, 1$):
$$\mathbf{h}_{\text{bw}}(t),\; \mathbf{u}_{\text{bw}}(t) = \text{LSTMSNPCell}_{\text{bw}}\bigl(\mathbf{x}(t),\; \mathbf{u}_{\text{bw}}(t{+}1)\bigr)$$

### 10.2 Output Concatenation

$$\mathbf{h}_{\text{concat}} = \bigl[\mathbf{h}_{\text{fw}}(T) \;\|\; \mathbf{h}_{\text{bw}}(1)\bigr] \in \mathbb{R}^{2H}$$

$$\hat{y} = \mathbf{w}_{\text{out}}^{\top}\,\mathbf{h}_{\text{concat}} + b_{\text{out}}$$

| Component | Specification |
|-----------|--------------|
| Forward Cell | `LSTMSNPCell` with consumption bias = 1.0 |
| Backward Cell | `LSTMSNPCell` with consumption bias = 1.0 |
| Input dimension | $d_{\text{in}} = 1$ |
| Hidden units | $H = 8$ per direction |
| Output layer | `Linear(16, 1)` |
| Stateful | Yes — both $\mathbf{u}_{\text{fw}}$ and $\mathbf{u}_{\text{bw}}$ persist |

---

## 11. LSTM‑SNP (Sequence‑over‑Lag)

**Folder:** [`LSTM_SNP/`](file:///tmp/Fuzzy-LSTM_SNP-Test_LAST60/LSTM_SNP)

This is the standard unidirectional LSTM‑SNP model identical to Type 1, but designed for **multi‑step lag experiments** (lag ∈ {1, 5, 10, 20, 30}).

### Architecture

For a lag‑$L$ experiment, the input is a sequence $\mathbf{x}(t), \mathbf{x}(t{+}1), \ldots, \mathbf{x}(t{+}L{-}1)$, processed sequentially:

$$\text{for } \tau = 0, 1, \ldots, L{-}1: \quad \mathbf{h}(\tau),\; \mathbf{u}(\tau) = \text{LSTMSNPCell}\bigl(\mathbf{x}(t{+}\tau),\; \mathbf{u}(\tau{-}1)\bigr)$$

$$\hat{y} = \mathbf{w}_{\text{out}}^{\top}\,\mathbf{h}(L{-}1) + b_{\text{out}}$$

| Component | Specification |
|-----------|--------------|
| Cell | `LSTMSNPCell` (unmodified, consumption bias = 1.0) |
| Input dimension | $d_{\text{in}} = 1$ |
| Sequence length | $L \in \{1, 5, 10, 20, 30\}$ |
| Hidden units | $H = 8$ |
| Output layer | `Linear(8, 1)` — applied to the **last** hidden state |

---

## 12. SNN Transformer (Spike‑Driven Transformer)

**Folder:** [`SNN_Transformers/`](file:///tmp/Fuzzy-LSTM_SNP-Test_LAST60/SNN_Transformers)

> [!IMPORTANT]
> This model adapts the Spike‑Driven Transformer (Yao et al.) for univariate time series forecasting. It replaces the standard attention mechanism with spike‑driven self‑attention (SDSA) achieving $O(ND)$ complexity instead of $O(N^2D)$.

### 12.1 Leaky Integrate‑and‑Fire (LIF) Neuron

The fundamental spiking neuron used throughout the model:

$$\mathbf{U}[t] = \mathbf{H}[t{-}1] + \mathbf{X}[t] \qquad \text{(Eq 1: membrane potential)}$$

$$\mathbf{S}[t] = \Theta(\mathbf{U}[t] - u_{\text{th}}) \qquad \text{(Eq 2: spike generation)}$$

$$\mathbf{H}[t] = V_{\text{reset}} \cdot \mathbf{S}[t] + (\beta \cdot \mathbf{U}[t]) \cdot (1 - \mathbf{S}[t]) \qquad \text{(Eq 3: temporal output)}$$

where:
- $\Theta(\cdot)$ is the Heaviside step function
- $u_{\text{th}} = 0.5$ is the firing threshold
- $\beta = 0.5$ is the decay factor
- $V_{\text{reset}} = 0.0$ is the reset potential
- $\mathbf{S}[t] \in \{0, 1\}$ are binary spikes

**Surrogate gradient** for backpropagation through the non‑differentiable Heaviside:

$$\frac{\partial \mathbf{S}}{\partial \mathbf{U}} \approx \frac{1}{(1 + \alpha|\mathbf{U} - u_{\text{th}}|)^2}, \qquad \alpha = 2.0$$

### 12.2 Spike‑Driven Self‑Attention (SDSA)

Given spike input $\mathbf{S} \in \{0,1\}^{B \times T \times D}$:

**Step 1 — Linear projections (float‑point):**
$$\mathbf{Q} = \mathbf{S}\,\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{S}\,\mathbf{W}_K, \quad \mathbf{V} = \mathbf{S}\,\mathbf{W}_V$$

**Step 2 — Convert to spike tensors:**
$$\mathbf{Q}_S = \text{SN}(\mathbf{Q}), \quad \mathbf{K}_S = \text{SN}(\mathbf{K}), \quad \mathbf{V}_S = \text{SN}(\mathbf{V})$$

where SN is the LIF neuron (Section 12.1).

**Step 3 — Hadamard product (element‑wise masking):**
$$\mathbf{QK} = \mathbf{Q}_S \odot \mathbf{K}_S \in \{0,1\}^{B \times T \times D}$$

> [!NOTE]
> Since spikes are binary, the Hadamard product $\mathbf{Q}_S \odot \mathbf{K}_S$ is equivalent to a logical AND mask — a spike passes only if both Q and K fire.

**Step 4 — Column sum (temporal aggregation):**
$$\mathbf{A} = \sum_{t=1}^{T} \mathbf{QK}_{:, t, :} \in \mathbb{R}^{B \times 1 \times D}$$

**Step 5 — Attention mask via SN:**
$$\mathbf{M} = \text{SN}(\mathbf{A}) \in \{0,1\}^{B \times 1 \times D}$$

**Step 6 — Gated output:**
$$\text{SDSA}(\mathbf{S}) = \mathbf{W}_{\text{proj}}\bigl(\mathbf{M} \odot \mathbf{V}_S\bigr)$$

### 12.3 Spiking Transformer Block (Membrane Shortcuts)

$$\mathbf{U}'_l = \text{SDSA}(\mathbf{S}_{l-1}) + \mathbf{U}_{l-1} \qquad \text{(Eq 9: membrane shortcut on SDSA)}$$

$$\mathbf{S}'_l = \text{SN}(\mathbf{U}'_l) \qquad \text{(Eq 10)}$$

$$\mathbf{S}_l = \text{SN}\bigl(\text{MLP}(\mathbf{S}'_l) + \mathbf{U}'_l\bigr) \qquad \text{(Eq 11: membrane shortcut on MLP)}$$

where the MLP is:
$$\text{MLP}(\mathbf{x}) = \mathbf{W}_2 \,\text{GELU}(\mathbf{W}_1\,\mathbf{x})$$
with $\mathbf{W}_1 \in \mathbb{R}^{D \times D_{\text{ff}}}$, $\mathbf{W}_2 \in \mathbb{R}^{D_{\text{ff}} \times D}$.

> [!TIP]
> The **Membrane Shortcut (MS)** connects membrane potentials (not spikes) between layers. This preserves the binary nature of spikes throughout the network, unlike the SEW shortcut which would add spikes to float‑point residuals.

### 12.4 Complete Model Architecture

$$\mathbf{u}_0 = \text{Linear}_{d_\text{in} \to D}(\mathbf{x}) \qquad \text{(input projection)}$$

$$\mathbf{U}_0 = \text{repeat}(\mathbf{u}_0, T) \in \mathbb{R}^{B \times T \times D} \qquad \text{(temporal expansion)}$$

$$\mathbf{S}_0 = \text{SN}(\mathbf{U}_0) \qquad \text{(Eq 8: initial spikes)}$$

$$\mathbf{S}_L, \mathbf{U}_L = \text{SpikingTransformerBlock}^L(\mathbf{S}_0, \mathbf{U}_0) \qquad \text{(L transformer blocks)}$$

$$\hat{y} = \text{Linear}_{D \to 1}\!\Bigl(\frac{1}{T}\sum_{t=1}^{T}\mathbf{S}_{L,:,t,:}\Bigr) \qquad \text{(Eq 12: GAP + regression head)}$$

| Hyperparameter | Value |
|---------------|-------|
| $D$ (d_model) | 8 |
| $n_{\text{heads}}$ | 2 |
| $L$ (n_layers) | 1 |
| $D_{\text{ff}}$ (ff_dim) | 16 |
| $T$ (spiking timesteps) | 4 |
| $\beta$ (LIF decay) | 0.5 |
| $u_{\text{th}}$ (firing threshold) | 0.5 |

---

## 13. Attention (Pure LSTM with Lag Experiments)

**Folder:** [`Attention/`](file:///tmp/Fuzzy-LSTM_SNP-Test_LAST60/Attention)

> [!NOTE]
> Despite the folder name, this model uses the **Pure LSTM** architecture (Section 8) — i.e., `nn.LSTM` — not a custom attention mechanism. The experiments in this folder vary the **lag** (sequence length) parameter across $L \in \{1, 5, 10, 20, 30\}$ to study the effect of input window size on standard LSTM performance.

| Component | Specification |
|-----------|--------------|
| Cell | `nn.LSTM` (standard PyTorch LSTM) |
| Input dimension | $d_{\text{in}} = 1$ |
| Sequence length | $L \in \{1, 5, 10, 20, 30\}$ |
| Hidden units | $H = 8$ |
| Output layer | `Linear(8, 1)` — applied to the **last** hidden state |
| Stateful | Yes |

---

## Summary Comparison

| Model | Cell Type | Gates | Input Dim | Output Layer | Fuzzy Component |
|-------|----------|-------|-----------|-------------|----------------|
| **Type 1** | LSTM‑SNP | hardsigmoid | 1 | Linear | None (baseline) |
| **Type 2** | LSTM‑SNP | hardsigmoid | 2 | Linear | Preprocessing (fixed FIS) |
| **Type 3** | Fuzzy LSTM‑SNP | Fuzzy (clamp) | 1 | Linear | Gate replacement (trainable) |
| **Type 3 Sigmoid** | Fuzzy LSTM‑SNP | Fuzzy (sigmoid) | 1 | Linear | Gate replacement (trainable) |
| **Type 4** | LSTM‑SNP | hardsigmoid | 1 | Fuzzy | Output layer (trainable FIS) |
| **Type 5** | Fuzzy LSTM‑SNP | Fuzzy (clamp) | 1 | Linear | Gate replacement (trainable) |
| **Pure LSTM** | Standard LSTM | sigmoid | 1 | Linear | None |
| **Pure GRU** | Standard GRU | sigmoid | 1 | Linear | None |
| **BiLSTM‑SNP** | 2× LSTM‑SNP | hardsigmoid | 1 | Linear | None |
| **LSTM‑SNP (Lag)** | LSTM‑SNP | hardsigmoid | 1 | Linear | None |
| **SNN Transformer** | LIF + SDSA | Spike‑driven | 1 | Linear | None |
| **Attention (LSTM Lag)** | Standard LSTM | sigmoid | 1 | Linear | None |

---

## Common Experimental Setup

All models share the following experimental protocol:

| Parameter | Value |
|-----------|-------|
| **Datasets** | Dow Jones Industrial Average, Monthly Lake Erie Levels, Monthly Milk Production, S&P 500 |
| **Preprocessing** | First‑order differencing → MinMaxScaler to $[-1, 1]$ |
| **Train/Test split** | Last segment used as test set |
| **Hidden units** | $H = 8$ (all models) |
| **Optimizer** | Adam ($\eta = 0.001$) |
| **Loss** | MSE |
| **Max epochs** | 100 |
| **Early stopping** | Patience = 10 (on validation RMSE) |
| **Gradient clipping** | $\|\nabla\|_{\max} = 1.0$ |
| **Runs** | 60 independent runs per configuration |
| **BPTT** | Truncated (state detached after each sample) |
| **Noise robustness** | Gaussian noise at $\{0.5\%, 5\%, 10\%, 15\%\}$ of input std, 10 draws per run |

### Evaluation Metrics

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$$

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

$$\text{NMSE} = \frac{\text{MSE}}{\|\mathbf{y} - \bar{y}\|_2^2}$$

where predictions are inverse‑transformed back to the original scale before metric computation.
