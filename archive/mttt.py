import math
import jax
import flax
import jax.numpy as jnp

from flax import linen as nn
from tqdm import tqdm
from jax import vjp, custom_vjp, grad, vmap
from typing import Any
from functools import partial
from jax.tree_util import tree_map

# Hyperparameters
n_layer = 12
n_embd = 768
n_head = 6
inner_net = "mlp_1" # Alternatively, use mlp_2

inner_chunk_size = 1
ctx_len = 2048

bias = False
decoder_LN = True

ilr = 1.0
BS = 1

key = jax.random.PRNGKey(0)

# Supporting Layers
class DummyLinearLayer(nn.Module):
  width: int
  use_bias: bool
  name: str

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.width, use_bias=self.use_bias, name=self.name)(x)
    return x

class DummyLayerNorm(nn.Module):
  name: str

  @nn.compact
  def __call__(self, x):
    x = nn.LayerNorm(name=self.name)(x)
    return x

class DummyNoOp(nn.Module):
  @nn.compact
  def __call__(self, x):
    return x

class TTTEncoder(nn.Module):
    mlp_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        if "mlp_1" in inner_net:
            y = nn.Dense(self.mlp_dim, use_bias=False, name="inner_Dense_0",
                      dtype=self.dtype)(x)
        elif "mlp_2" in inner_net:
            y = nn.Dense(int(self.mlp_dim * 4), use_bias=True, name="inner_Dense_0",
                      dtype=self.dtype)(x)
            y = nn.gelu(y)
            y = nn.Dense(self.mlp_dim, use_bias=True, name="inner_Dense_1",
                      dtype=self.dtype)(y)
        else:
            raise NotImplementedError("Inner Net %s Not Implemented." % (self.config.inner_net))
        
        return y

# MTTT Module
class MTTT(nn.Module):
    config: Any = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.width = n_embd
        self.num_heads = n_head
        self.inner_chunk_size = inner_chunk_size
        self.bias = bias
        self.use_decoder_LN = decoder_LN
        self.n_layer = n_layer

        # Initialize self-supervised task and encoder.
        self.psi = DummyLinearLayer(width=self.width // self.num_heads, use_bias=self.bias, name="psi")
        psi_params = self.psi.init(jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]

        self.phi = DummyLinearLayer(width=self.width // self.num_heads, use_bias=self.bias, name="phi")
        phi_params = self.phi.init(jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]

        self.g = DummyLinearLayer(width=self.width, use_bias=False, name="g")
        g_params = self.g.init(jax.random.PRNGKey(0), jnp.ones([1, self.width // self.num_heads]))["params"]

        self.h = DummyLinearLayer(width=self.width, use_bias=False, name="h")
        h_params = self.h.init(jax.random.PRNGKey(0), jnp.ones([1, self.width // self.num_heads]))["params"]

        self.encoder = TTTEncoder(mlp_dim=self.width // self.num_heads)
        encoder_params = self.encoder.init(jax.random.PRNGKey(0), jnp.ones([1, self.width // self.num_heads]))["params"]

        def get_multi_head_params(params, kernel_init="normal", std=0.02):
            flat_params = flax.traverse_util.flatten_dict(params, sep="[SEP]")
            for k in flat_params.keys():
                new_shape = (self.num_heads, *flat_params[k].shape)
                if 'scale' in k:
                    # initialize scale to 1
                    p = self.param(k, jax.nn.initializers.ones, new_shape, self.dtype)
                elif "kernel" in k:
                    if kernel_init == "normal":
                        if "inner_Dense_1" in k:
                            std = std / math.sqrt(2 * self.n_layer)
                        initializer = nn.initializers.normal(std)
                    elif kernel_init == "zero":
                        initializer = nn.initializers.zeros
                    else:
                        raise NotImplementedError("Initializer %s Not Implemented." % (kernel_init))
                    p = self.param(k, initializer, new_shape, self.dtype)
                else:
                    # initialize bias to 0
                    p = self.param(k, jax.nn.initializers.zeros, new_shape, self.dtype)
                flat_params[k] = p
            params_init = flax.traverse_util.unflatten_dict(flat_params, sep="[SEP]")
            return params_init

        self.encoder_params = get_multi_head_params(encoder_params, "normal")
        self.psi_params = get_multi_head_params(psi_params, "normal")
        self.phi_params = get_multi_head_params(phi_params, "normal")
        self.g_params = get_multi_head_params(g_params, "normal")
        self.h_params = get_multi_head_params(h_params, "normal", std=0.02 / math.sqrt(2 * self.n_layer))

        if self.use_decoder_LN:
            self.decoder_LN = DummyLayerNorm()
            decoder_LN_params = self.decoder_LN.init(jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]
        else:
            self.decoder_LN = DummyNoOp()
            decoder_LN_params = {}
        self.decoder_LN_params = get_multi_head_params(decoder_LN_params, "layer_norm")

    def __call__(self, batch):

        def f(phi_params, encoder_params, g_params, decoder_LN_params,
              psi_params, h_params, batch):

            def inner_loss(phi_params, g_params, decoder_LN_params, encoder_params, token):
                inner_input = self.phi.apply({"params": phi_params}, token)
                inner_input_transformed = self.encoder.apply({"params": encoder_params}, inner_input)
                inner_output = self.g.apply({"params": g_params}, inner_input_transformed)
                inner_output = self.decoder_LN.apply({"params": decoder_LN_params}, inner_output)
                loss = 0.5 * ((inner_output - token) ** 2).mean() * self.num_heads  # normalizer = N * d / H
                return loss, (0, 0)
                
            inner_loss_value_and_grad = jax.value_and_grad(inner_loss, argnums=3, has_aux=True)

            # Parallelize over batch.
            @vmap
            def update_embed(sequence):
                
                # Parallelize over head.
                @partial(vmap, axis_name="head")
                def parallelize_over_heads(psi_params, phi_params, encoder_params, g_params, decoder_LN_params, h_params, sequence_head):

                    @partial(jax.checkpoint, prevent_cse=False) # Use naive gradient checkpointing to avoid OOM
                    def compute_chunk(W_init_chunk, inputs):

                        @vmap
                        def forward(Wt, xt):
                            zt = self.psi.apply({"params": psi_params}, xt)
                            zt = self.encoder.apply({"params": Wt}, zt)
                            zt = self.h.apply({"params": h_params}, zt)
                            return zt

                        @vmap
                        def avg_G(G_cumsum, idx):
                            return tree_map(lambda g: g / idx, G_cumsum)

                        token_chunk = inputs['token_chunk']
                        token_idx = inputs['token_idx_chunk']

                        _, dldW_inner_chunk = vmap(partial(inner_loss_value_and_grad, phi_params, g_params, decoder_LN_params, W_init_chunk))(token_chunk) # Take gradients in parallel
                        G_cumsum = tree_map(partial(jnp.cumsum, axis=0), dldW_inner_chunk) # Cumulative sum of gradients
                        G_avg = avg_G(G_cumsum, token_idx)  # Average gradient based on idx

                        # Gradient descent function
                        def update_Wt(G, W_init):
                            Wt = tree_map(lambda p, g: p - ilr * g, W_init, G)
                            return Wt

                        cumulative_Wt = vmap(partial(update_Wt, W_init=W_init_chunk))(G_avg) # Take gradient step in parallel

                        W_last = tree_map(lambda x: x[-1], cumulative_Wt)
                        zt_chunk = forward(cumulative_Wt, token_chunk) # Forward through updated model for each token in parallel

                        return W_last, zt_chunk

                    # Prepare inputs for scan over tokens in sequence
                    sequence_chunked = sequence_head.reshape(-1, self.inner_chunk_size, self.width)
                    token_idx = jnp.arange(1, self.inner_chunk_size + 1, dtype=jnp.float32)
                    token_idx_chunked = jnp.tile(token_idx, (sequence_head.shape[0] // self.inner_chunk_size, 1))
                    inputs = {
                        'token_chunk': sequence_chunked,
                        'token_idx_chunk': token_idx_chunked,
                    }
                    
                    _, z_batch = jax.lax.scan(compute_chunk, encoder_params, inputs)
                    return z_batch.reshape(-1, self.width)

                sequence_mh = jnp.repeat(jnp.expand_dims(sequence, axis=0), repeats=self.num_heads, axis=0) # Duplicate input for multi-head
                embed_new = parallelize_over_heads(psi_params, phi_params, encoder_params, g_params, decoder_LN_params, h_params, sequence_mh)
                
                return embed_new.sum(axis=0)

            return update_embed(batch)

        return f(self.phi_params, self.encoder_params, self.g_params, self.decoder_LN_params,
                 self.psi_params, self.h_params, batch)

# Forward Method
def forward():
    mttt_layer = MTTT()
    mttt_layer_params = mttt_layer.init(key, jnp.ones([1, 1, 768], dtype=jnp.int32))['params'] # Initialize with random parameters.

    @jax.jit
    def fwd_fn(data):
        z = mttt_layer.apply({'params': mttt_layer_params}, data)
        return z

    data = jax.random.normal(key, (BS, ctx_len, n_embd), dtype=jnp.float32)

    print("** Forward Method **")
    for _ in tqdm(range(10)):
        output = fwd_fn(data)

# Backward Method
def backward():
    mttt_layer = MTTT()
    mttt_layer_params = mttt_layer.init(key, jnp.ones([1, 1, 768], dtype=jnp.int32))['params'] # Initialize with random parameters.

    def make_fwd_fn(model, inputs):
        
        @jax.jit
        def fwd_fn(params):
            z = model.apply({'params': params}, inputs)
            return z
        
        return fwd_fn
    
    data = jax.random.normal(key, (BS, ctx_len, n_embd), dtype=jnp.float32)
    fwd_fn = make_fwd_fn(mttt_layer, data)
    output, mttt_layer_vjp_fn = jax.vjp(fwd_fn, mttt_layer_params)

    print("** Backward Method **")
    for _ in tqdm(range(10)):
        gradient = mttt_layer_vjp_fn(data)
        
if __name__ == "__main__":
    forward()
    backward()
