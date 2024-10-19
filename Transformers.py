import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


class Transformer:
    def __init__(self, d_model, n_heads, num_encoder_layers, num_decoder_layers, d_ff, input_vocab_size, target_vocab_size, max_seq_length):
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_seq_length = max_seq_length

        # Initialisation des couches
        self.encoder_layers = [EncoderLayer(
            d_model, n_heads, d_ff) for _ in range(num_encoder_layers)]
        self.decoder_layers = [DecoderLayer(
            d_model, n_heads, d_ff) for _ in range(num_decoder_layers)]

        # Matrices d'embedding
        self.input_embedding = Embedding(input_vocab_size, d_model)
        self.target_embedding = Embedding(target_vocab_size, d_model)

        # Encodage positionnel
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Couche linéaire finale
        self.output_linear = Linear(d_model, target_vocab_size)

        # Liste de tous les paramètres pour l'optimisation
        self.parameters = []
        self._gather_parameters()

        # Initialisation des moments pour Adam
        self.optim_configs = {}
        for param in self.parameters:
            self.optim_configs[id(param)] = {'m': np.zeros_like(
                param.data), 'v': np.zeros_like(param.data)}

    def _gather_parameters(self):
        for layer in self.encoder_layers:
            self.parameters.extend(layer.get_parameters())
        for layer in self.decoder_layers:
            self.parameters.extend(layer.get_parameters())
        self.parameters.extend(self.input_embedding.get_parameters())
        self.parameters.extend(self.target_embedding.get_parameters())
        self.parameters.extend(self.output_linear.get_parameters())

    def forward(self, src_seq, tgt_seq_input):
        # Embedding et ajout de l'encodage positionnel
        src_embed = self.input_embedding.forward(
            src_seq) + self.positional_encoding.forward(src_seq.shape[1])
        tgt_embed = self.target_embedding.forward(
            tgt_seq_input) + self.positional_encoding.forward(tgt_seq_input.shape[1])

        # Encodeur
        encoder_output = src_embed
        for layer in self.encoder_layers:
            encoder_output = layer.forward(encoder_output)

        # Décodeur
        decoder_output = tgt_embed
        for layer in self.decoder_layers:
            decoder_output = layer.forward(decoder_output, encoder_output)

        # Couche linéaire finale
        output = self.output_linear.forward(decoder_output)
        output = softmax(output)
        return output

    def backward(self, d_output):
        # Rétropropagation à travers la couche linéaire finale
        d_decoder_output = self.output_linear.backward(d_output)

        # Rétropropagation à travers les couches du décodeur
        d_encoder_output = 0  # Initialisation des gradients pour l'encodeur
        for layer in reversed(self.decoder_layers):
            d_decoder_output, d_enc_output = layer.backward(d_decoder_output)
            d_encoder_output += d_enc_output  # Accumulation des gradients

        # Rétropropagation à travers les embeddings de la cible
        self.target_embedding.backward(d_decoder_output)

        # Rétropropagation à travers les couches de l'encodeur
        for layer in reversed(self.encoder_layers):
            d_encoder_output = layer.backward(d_encoder_output)

        # Rétropropagation à travers les embeddings de la source
        self.input_embedding.backward(d_encoder_output)

    def step(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Mise à jour des paramètres avec Adam
        for param in self.parameters:
            config = self.optim_configs[id(param)]
            config['m'] = beta1 * config['m'] + (1 - beta1) * param.grad
            config['v'] = beta2 * config['v'] + (1 - beta2) * (param.grad ** 2)

            m_unbiased = config['m'] / (1 - beta1)
            v_unbiased = config['v'] / (1 - beta2)

            param.data -= learning_rate * m_unbiased / \
                (np.sqrt(v_unbiased) + epsilon)


class EncoderLayer:
    def __init__(self, d_model, n_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

    def forward(self, x):
        # Self-attention avec connexion résiduelle et normalisation
        attn_output = self.self_attn.forward(x, x, x)
        x2 = x + attn_output
        x_norm1 = self.norm1.forward(x2)

        # Feed-forward avec connexion résiduelle et normalisation
        ffn_output = self.ffn.forward(x_norm1)
        x3 = x_norm1 + ffn_output
        x_norm2 = self.norm2.forward(x3)
        self.output = x_norm2
        return x_norm2

    def backward(self, d_output):
        # Rétropropagation à travers la deuxième normalisation
        d_norm2 = self.norm2.backward(d_output)

        # Rétropropagation à travers la connexion résiduelle après le feed-forward
        d_ffn = d_norm2.copy()
        d_residual2 = d_norm2.copy()

        # Rétropropagation à travers le feed-forward
        d_ffn = self.ffn.backward(d_ffn)

        # Somme des gradients de la connexion résiduelle et du feed-forward
        d_post_ffn = d_ffn + d_residual2

        # Rétropropagation à travers la première normalisation
        d_norm1 = self.norm1.backward(d_post_ffn)

        # Rétropropagation à travers la connexion résiduelle après l'attention
        d_attn = d_norm1.copy()
        d_residual1 = d_norm1.copy()

        # Rétropropagation à travers l'attention
        d_attn = self.self_attn.backward(d_attn)

        # Somme des gradients de la connexion résiduelle et de l'attention
        d_input = d_attn + d_residual1

        return d_input

    def get_parameters(self):
        params = []
        params.extend(self.self_attn.get_parameters())
        params.extend(self.ffn.get_parameters())
        params.extend(self.norm1.get_parameters())
        params.extend(self.norm2.get_parameters())
        return params


class DecoderLayer:
    def __init__(self, d_model, n_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)

    def forward(self, x, enc_output):
        # Self-attention avec masquage
        attn_output = self.self_attn.forward(x, x, x, mask='future')
        x = self.norm1.forward(x + attn_output)
        # Attention encodeur-décodeur
        attn_output = self.enc_dec_attn.forward(x, enc_output, enc_output)
        x = self.norm2.forward(x + attn_output)
        # Réseau feed-forward
        ffn_output = self.ffn.forward(x)
        x = self.norm3.forward(x + ffn_output)
        self.output = x
        return x

    def backward(self, d_output):
        # Rétropropagation à travers la troisième normalisation
        d_norm3 = self.norm3.backward(d_output)

        # Rétropropagation à travers la connexion résiduelle après le feed-forward
        d_ffn = d_norm3.copy()
        d_residual3 = d_norm3.copy()

        # Rétropropagation à travers le feed-forward
        d_ffn = self.ffn.backward(d_ffn)

        # Somme des gradients de la connexion résiduelle et du feed-forward
        d_post_ffn = d_ffn + d_residual3

        # Rétropropagation à travers la deuxième normalisation
        d_norm2 = self.norm2.backward(d_post_ffn)

        # Rétropropagation à travers la connexion résiduelle après l'attention encodeur-décodeur
        d_enc_dec_attn = d_norm2.copy()
        d_residual2 = d_norm2.copy()

        # Rétropropagation à travers l'attention encodeur-décodeur
        d_enc_dec_attn = self.enc_dec_attn.backward(d_enc_dec_attn)

        # Somme des gradients de la connexion résiduelle et de l'attention encodeur-décodeur
        d_post_enc_dec_attn = d_enc_dec_attn + d_residual2

        # Rétropropagation à travers la première normalisation
        d_norm1 = self.norm1.backward(d_post_enc_dec_attn)

        # Rétropropagation à travers la connexion résiduelle après la self-attention
        d_self_attn = d_norm1.copy()
        d_residual1 = d_norm1.copy()

        # Rétropropagation à travers la self-attention
        d_self_attn = self.self_attn.backward(d_self_attn)

        # Somme des gradients de la connexion résiduelle et de la self-attention
        d_input = d_self_attn + d_residual1

        # Les gradients par rapport à l'encodeur sont accumulés
        # À implémenter dans MultiHeadAttention
        d_encoder = self.enc_dec_attn.d_encoder_output

        return d_input, d_encoder

    def get_parameters(self):
        params = []
        params.extend(self.self_attn.get_parameters())
        params.extend(self.enc_dec_attn.get_parameters())
        params.extend(self.ffn.get_parameters())
        params.extend(self.norm1.get_parameters())
        params.extend(self.norm2.get_parameters())
        params.extend(self.norm3.get_parameters())
        return params


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.shape
        x = x.reshape(batch_size, seq_length, self.n_heads, self.d_k)
        x = x.transpose(0, 2, 1, 3)
        return x

    def combine_heads(self, x):
        x = x.transpose(0, 2, 1, 3)
        batch_size, seq_length, n_heads, d_k = x.shape
        x = x.reshape(batch_size, seq_length, n_heads * d_k)
        return x

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores + mask
        attn_weights = softmax(scores)
        output = np.matmul(attn_weights, V)
        self.attn_weights = attn_weights
        return output

    def forward(self, query, key, value, mask=None):
        Q = self.W_q.forward(query)
        K = self.W_k.forward(key)
        V = self.W_v.forward(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        if mask == 'future':
            seq_length = query.shape[1]
            mask = np.triu(
                np.ones((1, 1, seq_length, seq_length)) * -np.inf, k=1)
        else:
            mask = None

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = self.combine_heads(attn_output)

        output = self.W_o.forward(attn_output)
        self.output = output
        return output

    def backward(self, d_output):
        # Rétropropagation à travers W_o
        d_attn_output = self.W_o.backward(d_output)

        # Division des têtes
        d_attn_output = self.split_heads(d_attn_output)

        # Rétropropagation à travers l'attention
        d_Q, d_K, d_V = self._backward_attention(d_attn_output)

        # Combinaison des têtes
        d_Q = self.combine_heads(d_Q)
        d_K = self.combine_heads(d_K)
        d_V = self.combine_heads(d_V)

        # Rétropropagation à travers W_q, W_k, W_v
        d_query = self.W_q.backward(d_Q)
        d_key = self.W_k.backward(d_K)
        d_value = self.W_v.backward(d_V)

        return d_query, d_key, d_value

    def _backward_attention(self, d_output):
        # Calcul du gradient par rapport à Q, K, V
        # Simplification pour cet exemple (calculs exacts plus complexes)
        d_Q = d_output.copy()
        d_K = d_output.copy()
        d_V = d_output.copy()
        return d_Q, d_K, d_V

    def get_parameters(self):
        params = []
        params.extend(self.W_q.get_parameters())
        params.extend(self.W_k.get_parameters())
        params.extend(self.W_v.get_parameters())
        params.extend(self.W_o.get_parameters())
        return params


class PositionwiseFeedForward:
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x):
        self.input = x
        self.output = self.linear2.forward(relu(self.linear1.forward(x)))
        return self.output

    def backward(self, d_output):
        d_relu = self.linear2.backward(d_output)
        d_linear1 = self.linear1.backward(
            d_relu * relu_derivative(self.linear1.output))
        return d_linear1

    def get_parameters(self):
        params = []
        params.extend(self.linear1.get_parameters())
        params.extend(self.linear2.get_parameters())
        return params


class Embedding:
    def __init__(self, vocab_size, d_model):
        self.embedding_matrix = Parameter(
            np.random.randn(vocab_size, d_model) * 0.01)

    def forward(self, x):
        self.input = x
        self.output = self.embedding_matrix.data[x]
        return self.output

    def backward(self, d_output):
        np.add.at(self.embedding_matrix.grad, self.input, d_output)
        return None

    def get_parameters(self):
        return [self.embedding_matrix]


class PositionalEncoding:
    def __init__(self, d_model, max_seq_length):
        self.d_model = d_model
        self.positional_encoding = self.get_positional_encoding(max_seq_length)

    def get_positional_encoding(self, max_seq_length):
        position = np.arange(max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) *
                          (-np.log(10000.0) / self.d_model))
        pe = np.zeros((max_seq_length, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def forward(self, seq_length):
        return self.positional_encoding[:seq_length]


class Linear:
    def __init__(self, in_features, out_features):
        self.W = Parameter(np.random.randn(in_features, out_features) * 0.01)
        self.b = Parameter(np.zeros((1, out_features)))

    def forward(self, x):
        self.input = x  # Correction : stockage de l'entrée
        self.input_shape = x.shape
        batch_size, seq_length, _ = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1])
        output = np.dot(x_reshaped, self.W.data) + self.b.data
        self.output = output.reshape(batch_size, seq_length, -1)
        return self.output

    def backward(self, d_output):
        batch_size, seq_length, _ = d_output.shape
        d_output = d_output.reshape(-1, d_output.shape[-1])
        # Utilisation de self.input
        x = self.input.reshape(-1, self.input.shape[-1])
        self.W.grad += np.dot(x.T, d_output)
        self.b.grad += np.sum(d_output, axis=0, keepdims=True)
        d_input = np.dot(d_output, self.W.data.T)
        d_input = d_input.reshape(self.input_shape)
        return d_input

    def get_parameters(self):
        return [self.W, self.b]


class LayerNormalization:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = Parameter(np.ones((d_model,)))
        self.beta = Parameter(np.zeros((d_model,)))
        self.eps = eps

    def forward(self, x):
        self.input = x  # x a la forme (N, T, D)
        self.mean = x.mean(axis=-1, keepdims=True)  # (N, T, 1)
        self.std = x.std(axis=-1, keepdims=True)    # (N, T, 1)
        self.x_norm = (x - self.mean) / (self.std + self.eps)  # (N, T, D)
        self.output = self.gamma.data * self.x_norm + self.beta.data  # Diffusion sur D
        return self.output

    def backward(self, d_output):
        N, T, D = self.input.shape
        x_mu = self.input - self.mean
        std_inv = 1. / (self.std + self.eps)

        dx_norm = d_output * self.gamma.data  # (N, T, D)
        dvar = np.sum(dx_norm * x_mu * -0.5 * std_inv ** 3,
                      axis=-1, keepdims=True)  # (N, T, 1)
        dmean = np.sum(dx_norm * -std_inv, axis=-1, keepdims=True) + \
            dvar * np.mean(-2. * x_mu, axis=-1, keepdims=True)  # (N, T, 1)
        dx = (dx_norm * std_inv) + (dvar * 2 *
                                    x_mu / D) + (dmean / D)  # (N, T, D)

        # Mise à jour des gradients pour gamma et beta
        self.gamma.grad += np.sum(d_output * self.x_norm, axis=(0, 1))  # (D,)
        self.beta.grad += np.sum(d_output, axis=(0, 1))  # (D,)

        return dx

    def get_parameters(self):
        return [self.gamma, self.beta]


class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

# Classe d'aide pour gérer les paramètres


class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)
