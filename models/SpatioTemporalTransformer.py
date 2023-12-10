import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=300):
        """
        Initializes a positional embedding layer for each joint.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, src_seqs):
        """
        Implements the forward pass on the embedded joints.

        Args:
            src_seqs: A (batch_size, seq_len, dim) Tensor of sequences to process.

        Returns:
            The input sequence mutated to include the positional encodings.
        """
        src_seqs = src_seqs + self.pe[:, : src_seqs.shape[1], :]
        return self.dropout(src_seqs)


class JointEmbedding(nn.Module):
    def __init__(self, num_joints, joint_dim, embedding_dim, dropout, device):
        """
        Initializes a linear embedding layer for each joint.
        """
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList(
            [nn.Linear(joint_dim, embedding_dim) for _ in range(num_joints)]
        ).to(device)
        self.positional_embeddings = nn.ModuleList(
            [
                PositionalEncoding(embedding_dim, dropout=dropout, max_len=300)
                for _ in range(num_joints)
            ]
        ).to(device)

    def forward(self, src_seqs):
        """
        Method applies the linear joint embedding to each joint and concats them together.
        This maps from:
            (B, T, D) => (B, T, E)

        Where:
            1. B := Batch Size
            2. T := Sequence Length
            3. D := Raw Joint Feature Dimension
            4. E := Joint Embedding Dimension

        Args:
            src_seqs: A (batch_size, seq_len, dim) Tensor of sequences to process.

        Returns:
            A Tensor with the final dimension embedded.
        """
        pos_embeddings = []
        for i in range(self.num_joints):
            j_i = src_seqs[:, :, i * self.joint_dim : (i + 1) * self.joint_dim]
            e_i = self.embeddings[i](j_i)
            p_i = self.positional_embeddings[i](e_i)
            pos_embeddings.append(p_i)

        return torch.cat(pos_embeddings, dim=-1)


class SpatioTemporalAttention(nn.Module):
    def __init__(
        self,
        num_joints,
        seq_len,
        device,
        embedding_dim=128,
        ff_dim=256,
        num_heads=8,
        dropout=0.1,
        temporal_attention_horizon=120,
    ):
        """
        Initializes the ST Attention.
        """
        super().__init__()
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.input_dim = num_joints * embedding_dim
        self.num_heads = num_heads
        self.attention_dim = self.embedding_dim / num_heads
        assert self.attention_dim.is_integer()
        self.attention_dim = int(self.attention_dim)
        self.feedforward_dim = ff_dim
        self.device = device
        self.temporal_attention_horizon = temporal_attention_horizon

        # Attention Weights
        self.temporal_attention_weights = []
        self.spatial_attention_weights = []

        # Temporal Attention
        self.temporal_attention_q = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
                        for _ in range(num_joints)
                    ]
                )
                for _ in range(num_heads)
            ]
        ).to(self.device)
        self.temporal_attention_k = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
                        for _ in range(num_joints)
                    ]
                )
                for _ in range(num_heads)
            ]
        ).to(self.device)
        self.temporal_attention_v = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
                        for _ in range(num_joints)
                    ]
                )
                for _ in range(num_heads)
            ]
        ).to(self.device)
        self.temporal_head_projection = nn.Linear(self.input_dim, self.input_dim).to(self.device)
        self.temporal_tau = nn.Softmax(dim=-1).to(self.device)
        self.temporal_norm = nn.BatchNorm1d(self.input_dim).to(self.device)

        # Spatial Attention
        self.spatial_attention_q = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
                        for _ in range(num_joints)
                    ]
                )
                for _ in range(num_heads)
            ]
        ).to(self.device)
        self.spatial_attention_k = nn.ModuleList(
            [
                nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
                for _ in range(num_heads)
            ]
        ).to(self.device)
        self.spatial_attention_v = nn.ModuleList(
            [
                nn.Linear(self.embedding_dim, self.attention_dim, bias=False)
                for _ in range(num_heads)
            ]
        ).to(self.device)
        self.spatial_head_projection = nn.Linear(self.input_dim, self.input_dim).to(self.device)
        self.spatial_tau = nn.Softmax(dim=-1).to(self.device)
        self.spatial_norm = nn.BatchNorm1d(self.input_dim).to(self.device)

        # Setup layers
        self.attention_dim_norm = torch.sqrt(torch.tensor(self.attention_dim).float())
        self.dropout = nn.Dropout(p=dropout).to(self.device)

        # Feed Forward Layers
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, self.input_dim),
        ).to(self.device)
        self.fc_norm = torch.nn.BatchNorm1d(self.input_dim).to(self.device)

    def _generate_square_subsequent_mask(self, shape):
        """
        Implements a mask for the temporal attention block.

        Args:
            shape: The size of the mask.

        Returns:
            A Tensor with attention predictions.
        """
        mask = (torch.triu(torch.ones(shape, shape)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _temporal_attention(self, src_seqs):
        """
        Implements the temporal attention mechanism.

        Args:
            src_seqs: A (batch_size, seq_len, dim) Tensor of sequences to process.

        Returns:
            A Tensor with temporal attention values.
        """
        mask = self._generate_square_subsequent_mask(src_seqs.shape[1]).to(self.device)

        temporal_attentions = []
        for h in range(self.num_heads):
            temporal_joint_attentions = []

            for i in range(self.num_joints):
                # Slice sequence to the specific joint
                j_i = src_seqs[
                    :, :, i * self.embedding_dim : (i + 1) * self.embedding_dim
                ]

                # Setup Projections
                q_i = self.temporal_attention_q[h][i](j_i)
                k_i = self.temporal_attention_k[h][i](j_i)
                v_i = self.temporal_attention_v[h][i](j_i)

                # Compute Attention
                a_i = (q_i @ k_i.permute(0, 2, 1)) / self.attention_dim_norm
                a_i = self.temporal_tau(a_i + mask)
                attention_i = a_i @ v_i
                temporal_joint_attentions.append(attention_i)

            temporal_attentions.append(torch.cat(temporal_joint_attentions, dim=-1))

        # Combine attention head results
        temporal_attention = torch.cat(temporal_attentions, dim=-1)
        temporal_attention = self.temporal_head_projection(temporal_attention)
        return temporal_attention

    def _spatial_attention(self, src_seqs):
        """
        Implements the spatial attention mechanism.

        Args:
            src_seqs: A (batch_size, seq_len, dim) Tensor of sequences to process.

        Returns:
            A Tensor with spatial attention values.
        """
        batch_size, _, _ = src_seqs.shape

        spatial_attentions = []
        for h in range(self.num_heads):
            # Reshape to (batch_size, seq_len, num_joints, embedding_dimension)
            joint_embeddings = torch.reshape(
                src_seqs[:, :, :], (batch_size, self.seq_len, -1, self.embedding_dim)
            )

            # Setup Projections
            q_is = []
            for i in range(self.num_joints):
                # Each joint embedding has it's own projection
                q_i = self.spatial_attention_q[h][i](joint_embeddings[:, :, i, :])
                q_is.append(torch.reshape(q_i, (q_i.shape[0], q_i.shape[1], 1, q_i.shape[2])))

            q = torch.cat(q_is, dim=2)
            k = self.spatial_attention_k[h](joint_embeddings)
            v = self.spatial_attention_v[h](joint_embeddings)

            # Compute Attention
            a = (q @ k.permute(0, 1, 3, 2)) / self.attention_dim_norm
            a = self.spatial_tau(a)
            attention = torch.reshape(a @ v, (batch_size, self.seq_len, -1))
            spatial_attentions.append(attention)

        # Combine attention head results
        spatial_attention = torch.cat(spatial_attentions, dim=-1)
        spatial_attention = self.spatial_head_projection(spatial_attention)
        return spatial_attention

    def get_temporal_attention_weights(self):
        return self.temporal_attention_weights

    def get_spatial_attention_weights(self):
        return self.spatial_attention_weights

    def forward(self, src_seqs):
        """
        Implements the forward function for ST Attention.

        Args:
            src_seqs: A (batch_size, seq_len, dim) Tensor of sequences to process.

        Returns:
            A Tensor with attention predictions.
        """

        # 1. Compute Temporal Attention
        temporal_attention = self._temporal_attention(src_seqs)
        temporal_attention = self.dropout(temporal_attention) + src_seqs
        temporal_attention = self.temporal_norm(
            temporal_attention.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # 2. Compute Spatial Attention
        spatial_attention = self._spatial_attention(src_seqs)
        spatial_attention = self.dropout(spatial_attention) + src_seqs
        spatial_attention = self.spatial_norm(
            spatial_attention.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # 3. Compute Spatio-temporal Attention
        attention = temporal_attention + spatial_attention
        attention = self.dropout(self.fc(attention)) + attention
        attention = self.fc_norm(attention.permute(0, 2, 1)).permute(0, 2, 1)

        return attention


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        num_joints,
        joint_dim,
        seq_len,
        input_dim,
        device,
        embedding_dim=128,
        ff_dim=256,
        embedding_dropout=0.1,
        num_heads=8,
        attention_layers=8,
        temporal_attention_horizon=120,
    ):
        """
        Initializes the ST Transformer.
        """
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout
        self.attention_layers = attention_layers
        self.num_heads = num_heads
        self.device = device

        self.temporal_attention_weights_all_layers = []
        self.spatial_attention_weights_all_layers = []

        # Define Modules
        self.joint_embeddings = JointEmbedding(
            num_joints, joint_dim, embedding_dim, embedding_dropout, device
        )
        self.spatio_temporal_attention = nn.ModuleList(
            [
                SpatioTemporalAttention(
                    num_joints,
                    seq_len,
                    device,
                    embedding_dim,
                    ff_dim,
                    num_heads,
                    embedding_dropout,
                    temporal_attention_horizon,
                )
                for _ in range(attention_layers)
            ]
        ).to(self.device)
        self.output = nn.Linear(num_joints * embedding_dim, num_joints * joint_dim).to(self.device)

    def forward(self, src_seqs):
        """
        Implements the forward function for ST Transformer.

        Args:
            src_seqs: A (batch_size, seq_len, dim) Tensor of sequences to process.

        Returns:
            A Tensor with auto-regressive predictions.
        """
        embed_seqs = self.joint_embeddings(src_seqs)

        # Clear previous weights
        self.temporal_attention_weights_all_layers = []
        self.spatial_attention_weights_all_layers = []

        # Iterate over the spatio-temporal attention layers
        attention_seqs = embed_seqs

        for i in range(self.attention_layers):
            attention_seqs = self.spatio_temporal_attention[i](attention_seqs)

            if i == 0:
                self.temporal_attention_weights_all_layers.append(
                    self.spatio_temporal_attention[i].get_temporal_attention_weights()
                )
                self.spatial_attention_weights_all_layers.append(
                    self.spatio_temporal_attention[i].get_spatial_attention_weights()
                )

        output_seqs = self.output(attention_seqs)

        return output_seqs + src_seqs
