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
        src_seqs = src_seqs + self.pe[:, :src_seqs.shape[1], :]
        return self.dropout(src_seqs)


class JointEmbedding(nn.Module):
    def __init__(self, num_joints, joint_dim, embedding_dim, dropout):
        """
        Initializes a linear embedding layer for each joint.
        """
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList([nn.Linear(joint_dim, embedding_dim) for _ in range(num_joints)])
        self.positional_embeddings = nn.ModuleList(
            [PositionalEncoding(embedding_dim, dropout=dropout, max_len=300) for _ in range(num_joints)])

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
            j_i = src_seqs[:, :, i * self.joint_dim:(i+1) * self.joint_dim]
            e_i = self.embeddings[i](j_i)
            p_i = self.positional_embeddings[i](e_i)
            pos_embeddings.append(p_i)

        return torch.cat(pos_embeddings, dim=-1)


class SpatioTemporalAttention(nn.Module):
    def __init__(self, num_joints, seq_len, embedding_dim, num_heads=2, dropout=0.1):
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

        # Temporal Attention
        self.temporal_attention_q = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.embedding_dim, self.attention_dim, bias=False) for _ in range(num_joints)])
            for _ in range(num_heads)
        ])
        self.temporal_attention_k = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.embedding_dim, self.attention_dim, bias=False) for _ in range(num_joints)])
            for _ in range(num_heads)
        ])
        self.temporal_attention_v = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.embedding_dim, self.attention_dim, bias=False) for _ in range(num_joints)])
            for _ in range(num_heads)
        ])
        self.temporal_head_projection = nn.Linear(self.input_dim, self.input_dim)
        self.temporal_tau = nn.Softmax(dim=-1)
        self.temporal_norm = nn.BatchNorm1d(self.input_dim)

        # Spatial Attention
        self.spatial_attention_q = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.embedding_dim, self.attention_dim, bias=False) for _ in range(num_joints)])
            for _ in range(num_heads)
        ])
        self.spatial_attention_k = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.attention_dim, bias=False) for _ in range(num_heads)])
        self.spatial_attention_v = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.attention_dim, bias=False) for _ in range(num_heads)])
        self.spatial_head_projection = nn.Linear(self.input_dim, self.input_dim)
        self.spatial_tau = nn.Softmax(dim=-1)
        self.spatial_norm = nn.BatchNorm1d(self.input_dim)

        # Setup layers
        self.attention_dim_norm = torch.sqrt(torch.tensor(self.attention_dim).float())
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.input_dim, self.input_dim)
        self.fc_norm = torch.nn.BatchNorm1d(self.input_dim)

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
        mask = self._generate_square_subsequent_mask(src_seqs.shape[1])

        temporal_attentions = []
        for h in range(self.num_heads):
            temporal_joint_attentions = []

            for i in range(self.num_joints):
                # Slice sequence to the specific joint
                j_i = src_seqs[:, :, i * self.embedding_dim:(i+1) * self.embedding_dim]

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
            spatial_attentions_over_time = []

            for i in range(self.seq_len):
                # Slice sequence to the embedding at each timestep
                #   and reshape to (batch_size, num_joints, embedding_dimension)
                e_i = torch.reshape(src_seqs[:, i, :], (batch_size, -1, self.embedding_dim))

                # Setup Projections
                q_is = []
                for j in range(self.num_joints):
                    # Each joint embedding has it's own projection
                    q_ij = self.spatial_attention_q[h][j](e_i[:, j, :])
                    q_is.append(torch.reshape(q_ij, (q_ij.shape[0], 1, q_ij.shape[1])))

                q_i = torch.cat(q_is, dim=1)
                k_i = self.spatial_attention_k[h](e_i)
                v_i = self.spatial_attention_v[h](e_i)

                # Compute Attention
                a_i = (q_i @ k_i.permute(0, 2, 1)) / self.attention_dim_norm
                a_i = self.spatial_tau(a_i)
                attention_i = torch.reshape(a_i @ v_i, (batch_size, 1, -1))
                spatial_attentions_over_time.append(attention_i)

            spatial_attentions.append(torch.cat(spatial_attentions_over_time, dim=1))

        # Combine attention head results
        spatial_attention = torch.cat(spatial_attentions, dim=-1)
        spatial_attention = self.spatial_head_projection(spatial_attention)
        return spatial_attention

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
        temporal_attention = self.temporal_norm(temporal_attention.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. Compute Spatial Attention
        spatial_attention = self._spatial_attention(src_seqs)
        spatial_attention = self.dropout(spatial_attention) + src_seqs
        spatial_attention = self.spatial_norm(spatial_attention.permute(0, 2, 1)).permute(0, 2, 1)

        # 3. Compute Spatio-temporal Attention
        attention = temporal_attention + spatial_attention
        attention = self.dropout(self.fc(attention)) + attention
        attention = self.fc_norm(attention.permute(0, 2, 1)).permute(0, 2, 1)
        return attention


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_joints, joint_dim, seq_len, input_dim, embedding_dim, embedding_dropout, attention_layers=2):
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

        # Define Modules
        self.joint_embeddings = JointEmbedding(num_joints, joint_dim, embedding_dim, embedding_dropout)
        self.spatio_temporal_attention = nn.ModuleList(
            [SpatioTemporalAttention(num_joints, seq_len, embedding_dim) for _ in range(attention_layers)])
        self.output = nn.Linear(num_joints * embedding_dim, num_joints * joint_dim)

    def forward(self, src_seqs):
        """
        Implements the forward function for ST Transformer.

        Args:
            src_seqs: A (batch_size, seq_len, dim) Tensor of sequences to process.

        Returns:
            A Tensor with auto-regressive predictions.
        """
        embed_seqs = self.joint_embeddings(src_seqs)

        # Iterate over the spatio-temporal attention layers
        attention_seqs = embed_seqs
        for i in range(self.attention_layers):
            attention_seqs = self.spatio_temporal_attention[i](attention_seqs)

        output_seqs = self.output(attention_seqs)

        print(
            f"src_seqs.shape={src_seqs.shape}\n",
            f"attention_seqs.shape={attention_seqs.shape}\n",
            f"output_seqs.shape={output_seqs.shape}",
        )
        return output_seqs + src_seqs  # Residual connection
