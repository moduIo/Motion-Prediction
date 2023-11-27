import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
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
    def __init__(self, num_joints, embedding_dim, num_heads=1):
        """
        Initializes the ST Attention.
        """
        super().__init__()
        self.num_joints = num_joints
        self.embedding_dim = embedding_dim
        self.input_dim = num_joints * embedding_dim
        self.num_heads = num_heads
        self.attention_dim = self.input_dim / num_heads
        assert self.attention_dim.is_integer()

        # Setup layers
        self.fc = nn.Linear(self.input_dim, self.input_dim)

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

    def forward(self, src_seqs):
        """
        Implements the forward function for ST Attention.

        Args:
            src_seqs: A (batch_size, seq_len, dim) Tensor of sequences to process.

        Returns:
            A Tensor with attention predictions.
        """
        mask = self._generate_square_subsequent_mask(src_seqs.shape[1])
        print(mask.shape)
        print(mask)
        return self.fc(src_seqs)


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_joints, joint_dim, input_dim, embedding_dim, embedding_dropout):
        """
        Initializes the ST Transformer.
        """
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout

        # Define Modules
        self.joint_embeddings = JointEmbedding(num_joints, joint_dim, embedding_dim, embedding_dropout)
        self.st_attention = SpatioTemporalAttention(num_joints, embedding_dim)
        self.output = nn.Linear(num_joints * embedding_dim, num_joints * joint_dim)

    def forward(self, src_seqs):
        """
        Implements the forward function for ST Transformer.

        Args:
            src_seqs: A (batch_size, seq_len, dim) Tensor of sequences to process.

        Returns:
            A Tensor with auto-regressive predictions.
        """
        embeddings_seqs = self.joint_embeddings(src_seqs)
        attention_seqs = self.st_attention(embeddings_seqs)
        output_seqs = self.output(attention_seqs)
        print(
            f"src_seqs.shape={src_seqs.shape}",
            f"embeddings_seqs.shape={embeddings_seqs.shape}",
            f"attention_seqs.shape={attention_seqs.shape}",
            f"output_seqs.shape={output_seqs.shape}",
        )
        return output_seqs + src_seqs  # Residual connection
