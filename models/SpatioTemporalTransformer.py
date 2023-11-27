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

    def forward(self, x):
        """
        Implements the forward pass on the embedded joints.
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


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
            src_seqs: A Tensor of sequences to process.

        Returns:
            A Tensor with the final dimension embedded.
        """
        pos_embeddings = []
        for i in range(self.num_joints):
            j_i = src_seqs[:, :, i * self.joint_dim:(i+1) * self.joint_dim]
            e_i = self.embeddings[i](j_i)
            p_i = self.positional_embeddings[i](e_i)
            pos_embeddings.append(p_i)

        pos_seqs = torch.cat(pos_embeddings, dim=-1)
        return pos_seqs


class SpatioTemporalAttentionBlock(nn.Module):
    ...


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_joints, joint_dim, input_dim, embedding_dim, embedding_dropout):
        """
        """
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout

        # Define Modules
        self.joint_embedding = JointEmbedding(num_joints, joint_dim, embedding_dim, embedding_dropout)
        self.output = nn.Linear(num_joints * embedding_dim, num_joints * joint_dim)

    def forward(self, src_seqs):
        """
        """
        embeddings_seqs = self.joint_embedding(src_seqs)
        print(f"src_seqs.shape={src_seqs.shape}", f"embeddings_seqs.shape={embeddings_seqs.shape}")
        return self.output(embeddings_seqs)
