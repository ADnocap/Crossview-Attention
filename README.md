# Crossview-Attention
This project merges inverted variate-token Transformers with locality-biased attention masks to examine how different encoding layouts interact with causal decay patterns in forecasting tasks.

## Mathematical Framework

### Variate-Token Encoding

Given a multivariate time-series $X \in \mathbb{R}^{T \times D}$ where $T$ is the sequence length and $D$ is the number of variates, the framework explores two encoding perspectives:

**Standard Layout**: Each token represents a time step across all variates

$$\text{Token}_t = [x_{t,1}, x_{t,2}, \ldots, x_{t,D}]$$

**Inverted Layout**: Each token represents a single variate across time

$$\text{Token}_d = [x_{1,d}, x_{2,d}, \ldots, x_{T,d}]$$

### Locality-Biased Attention

The attention mechanism incorporates a locality bias through a decay function:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M_{\text{locality}}\right)V$$

where $M_{\text{locality}}$ is a mask that encodes temporal proximity:

$$M_{\text{locality}}[i,j] = -\lambda \cdot |i - j|$$

The parameter $\lambda$ controls the strength of the locality bias, with larger values enforcing stronger preference for nearby time steps.

### Causal Decay

The framework models information decay over time using:

$$w(t, \tau) = e^{-\alpha \cdot \tau}$$

where $\tau = |t - t'|$ is the temporal distance and $\alpha$ is the decay rate. This ensures that:
- Recent observations receive higher weight
- Distant past information contributes proportionally less
- Causal relationships are preserved (no future leakage)

### Crossview Integration

The crossview mechanism combines representations from both encoding layouts through a weighted fusion:

$$H_{\text{cross}} = \gamma \cdot H_{\text{standard}} + (1-\gamma) \cdot H_{\text{inverted}}$$

where $\gamma \in [0,1]$ balances the contribution of each view, allowing the model to leverage complementary patterns captured by different encoding schemes.

## The dual-view approach captures:

**Temporal Correlations** (Standard Layout): 

$$\text{Corr}_{\text{time}}(t_i, t_j) = \frac{\langle X_{t_i}, X_{t_j} \rangle}{\|X_{t_i}\| \|X_{t_j}\|}$$

**Variate Correlations** (Inverted Layout):

$$\text{Corr}_{\text{var}}(d_i, d_j) = \frac{\langle X_{:,d_i}, X_{:,d_j} \rangle}{\|X_{:,d_i}\| \|X_{:,d_j}\|}$$

Both perspectives provide complementary inductive biases that can improve forecasting performance.

### Effective Receptive Field

The combination of locality bias and multi-view encoding creates an effective receptive field:

$$\text{ERF} = \sum_{l=1}^{L} w_l \cdot \left(1 + e^{-\alpha \cdot l}\right)$$

where $L$ is the number of layers and $w_l$ is the window size at layer $l$. This allows the model to balance local precision with global context.

## Forecasting Objective

The model is trained to minimize prediction error over a forecasting horizon $H$:

$$\mathcal{L} = \frac{1}{H} \sum_{h=1}^{H} \|\hat{X}_{t+h} - X_{t+h}\|_2^2$$

For multi-step forecasting, the objective can be extended with a horizon-weighted loss:

$$\mathcal{L}_{\text{weighted}} = \sum_{h=1}^{H} w_h \cdot \|\hat{X}_{t+h} - X_{t+h}\|_2^2$$

where $w_h = \frac{1}{1 + \beta \cdot h}$ down-weights distant predictions, and $\beta$ controls the decay rate.
