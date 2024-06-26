import jax
import jax.numpy as jnp
import haiku as hk
import optax
import agents
import numpy as np


class ContrastiveAgent(agents.LearnedAgent):
    name = "contrastive"
    description = r"$\bf{contrastive\ planning\ (ours)}$"
    color = "#DC8243"

    def get_waypoint(self, s, g, n_wypt):
        sg = jnp.array([s, g])
        psi_s, psi_g = self(sg)

        d, c = self.repr_dim, self.c
        A = self.params["~"]["A"]
        I = jnp.eye(d)
        M = np.zeros((n_wypt * d, n_wypt * d))
        for i in range(n_wypt):
            M[d * i : d * (i + 1), d * i : d * (i + 1)] = (c + 1) / c * I + c / (
                c + 1
            ) * A.T @ A
            if i + 1 < n_wypt:
                M[d * i : d * (i + 1), d * (i + 1) : d * (i + 2)] = -A.T
                M[d * (i + 1) : d * (i + 2), d * i : d * (i + 1)] = -A
        M = jnp.array(M)

        if n_wypt == 1:
            eta = A @ psi_s + A.T @ psi_g
        else:
            eta = jnp.block(
                [A @ psi_s, jnp.zeros((n_wypt - 2) * self.repr_dim), A.T @ psi_g]
            )

        w_vec = jnp.linalg.solve(M, eta).reshape((-1, self.repr_dim))
        psi_w = w_vec
        return psi_w

    def __call__(self, x):
        phi, psi = self.repr_fn.apply(self.params, x)
        return psi
    
    def hamming_distance(self, x, y):
        """Compute the Hamming distance between two binary arrays."""
        return jnp.sum(x != y, axis=-1)
    
    def shortest_path_classifier_loss(self, x0, xT, x_neg):
        # Assuming x0, xT, and x_neg are batches of binary data
        batch_size = x0.shape[0]

        # Calculate positive and negative distances
        pos_distances = self.hamming_distance(x0, xT)  # (batch_size,)
        neg_distances = jnp.array([self.hamming_distance(x0, neg) for neg in x_neg])  # (batch_size, batch_size)

        # Convert distances to logits (negative distances for softmax cross-entropy)
        pos_logits = -pos_distances[:, None]  # (batch_size, 1)
        neg_logits = -neg_distances  # (batch_size, batch_size)
        
        # Concatenate positive and negative logits
        logits = jnp.concatenate([pos_logits, neg_logits], axis=1)  # (batch_size, batch_size + 1)
        
        # Convert logits to float
        logits = logits.astype(jnp.float32)
        
        # Compute the log-softmax for contrastive loss
        labels = jnp.zeros(batch_size, dtype=int)  # Assuming positive samples are labeled as 0
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, logits.shape[-1]))

        accuracy = jnp.mean(jnp.argmax(logits, axis=1) == labels)
        
        return loss.mean(), accuracy

    def loss_fn(self, params, x0, xT, x_neg):
        phi, _psi = self.repr_fn.apply(params, x0)
        _phi, psi = self.repr_fn.apply(params, xT)

        l2 = (jnp.mean(psi**2) + jnp.mean(_psi**2)) / 2
        I = jnp.eye(self.batch_size)
        l_align = jnp.sum((phi - psi) ** 2, axis=1)

        pdist = jnp.mean((phi[:, None] - psi[None]) ** 2, axis=-1)
        l_unif = (
            jax.nn.logsumexp(-(pdist * (1 - I)), axis=1)
            + jax.nn.logsumexp(-(pdist.T * (1 - I)), axis=1)
        ) / 2.0

        loss = l_align + l_unif

        accuracy = jnp.mean(jnp.argmin(pdist, axis=1) == jnp.arange(self.batch_size))
        dual_loss = params["log_lambda"] * (self.c - jax.lax.stop_gradient(l2))
        metrics = (l_unif.mean(), l_align.mean(), accuracy, l2)

        # calculate classifier loss
        x0 = jnp.array(x0)
        xT = jnp.array(xT)
        x_neg = jnp.array(x_neg)
        classifier_loss, classifier_accuracy = self.shortest_path_classifier_loss(x0, xT, x_neg)


        metrics = dict(
            loss=loss.mean(),
            l_unif=l_unif.mean(),
            l_align=l_align.mean(),
            accuracy=accuracy,
            l2=l2,
            classifier_loss=classifier_loss,
            classifier_accuracy=classifier_accuracy,
        )
        return (
            loss.mean()
            + jax.lax.stop_gradient(jnp.exp(params["log_lambda"])) * l2
            + dual_loss,
            metrics,
        )

    # def get_data(self, key, max_horizon=200):
    #     key, rng1, rng2 = jax.random.split(key, 3)
    #     i = jax.random.randint(
    #         rng1, shape=(self.batch_size,), minval=0, maxval=self.ds.size - 1
    #     )
    #     horizon = self.ds.next_end[i] - i
    #     probs = self.gamma ** jnp.tile(
    #         jnp.arange(max_horizon)[None], (self.batch_size, 1)
    #     )
    #     mask = jnp.arange(max_horizon)[None] <= horizon[:, None]
    #     probs *= mask
    #     probs /= jnp.sum(probs, axis=1, keepdims=1)
    #     log_probs = jnp.log(probs)
    #     delta = jax.random.categorical(key=rng2, logits=log_probs)
    #     s = self.ds.observations[i]
    #     ns = self.ds.observations[i + delta]
    #     return key, s, ns

    def get_data(self, key, max_horizon=200):
        key, rng1, rng2, rng3 = jax.random.split(key, 4)
        
        # Sample indices for positive pairs
        i = jax.random.randint(rng1, shape=(self.batch_size,), minval=0, maxval=self.ds.size - 1)
        
        # Calculate horizon for each sample
        horizon = self.ds.next_end[i] - i
        
        # Calculate probabilities for selecting positive pairs
        probs = self.gamma ** jnp.tile(jnp.arange(max_horizon)[None], (self.batch_size, 1))
        mask = jnp.arange(max_horizon)[None] <= horizon[:, None]
        probs *= mask
        probs /= jnp.sum(probs, axis=1, keepdims=1)
        log_probs = jnp.log(probs)
        
        # Sample delta values for positive pairs
        delta = jax.random.categorical(key=rng2, logits=log_probs)
        
        # Get anchor and positive samples
        s = self.ds.observations[i]
        ns = self.ds.observations[i + delta]
        
        # Sample indices for negative samples
        j = jax.random.randint(rng3, shape=(self.batch_size,), minval=0, maxval=self.ds.size - 1)
        # Ensure negative samples are from different trajectories
        j = j + (i == j) * (j + 1)
        
        # Get negative samples
        neg = self.ds.observations[j]
        
        return key, s, ns, neg

    def _repr_fn(self, x):
        dtype = x.dtype
        x = jax.nn.swish(hk.Linear(64)(x))
        for _ in range(5):
            delta = hk.Linear(64)(jax.nn.swish(x))
            x = jax.nn.swish(hk.Linear(64)(delta) + x)
        psi = hk.Linear(self.repr_dim)(x)

        A = hk.get_parameter(
            "A",
            shape=(self.repr_dim, self.repr_dim),
            dtype=dtype,
            init=hk.initializers.Identity(),
        )
        if self.use_rotation:
            Q = self.make_ortho(A)
            phi = psi @ Q.T
        else:
            phi = psi @ A.T
        return phi, psi

    def make_ortho(self, X):
        I = jnp.eye(X.shape[0])
        A = X - X.T
        return (I + A) @ jnp.linalg.inv(I - A)
