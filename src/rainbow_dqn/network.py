import flax.nnx as nnx
import jax
import jax.numpy as jnp


class NoisyLinear(nnx.Module):
    def __init__(self, in_features, out_features, rngs, sigma_init=0.5):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nnx.Param(
            jax.nn.initializers.lecun_normal()(
                rngs.params(), (in_features, out_features)
            )
        )
        self.weight_sigma = nnx.Param(jnp.full((in_features, out_features), sigma_init))
        self.bias_mu = nnx.Param(
            jax.nn.initializers.zeros(rngs.params(), (out_features,))
        )
        self.bias_sigma = nnx.Param(jnp.full((out_features,), sigma_init))

    def __call__(self, x: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        key_w, key_b = jax.random.split(key)
        weight_eps = jax.random.normal(key_w, (self.in_features, self.out_features))
        bias_eps = jax.random.normal(key_b, (self.out_features,))
        weight = self.weight_mu.value + self.weight_sigma.value * weight_eps
        bias = self.bias_mu.value + self.bias_sigma.value * bias_eps
        return jnp.dot(x, weight) + bias


class DuelingDQN(nnx.Module):
    def __init__(self, state_dim, num_actions, dense_size, rngs):
        self.adv1 = NoisyLinear(state_dim, dense_size, rngs)
        self.adv2 = NoisyLinear(dense_size, dense_size, rngs)
        self.adv_out = NoisyLinear(dense_size, num_actions, rngs)
        self.val1 = NoisyLinear(state_dim, dense_size, rngs)
        self.val2 = NoisyLinear(dense_size, dense_size, rngs)
        self.val_out = NoisyLinear(dense_size, 1, rngs)

    def __call__(self, x: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        keys = jax.random.split(key, 6)
        adv = nnx.relu(self.adv1(x, keys[0]))
        adv = nnx.relu(self.adv2(adv, keys[1]))
        adv = self.adv_out(adv, keys[2])
        val = nnx.relu(self.val1(x, keys[3]))
        val = nnx.relu(self.val2(val, keys[4]))
        val = self.val_out(val, keys[5])
        return val + (adv - jnp.mean(adv, axis=-1, keepdims=True))
