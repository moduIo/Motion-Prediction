import torch
import torch.nn as nn


class JointEmbedder(nn.Module):
    def __init__(self, num_joints, joint_dim, embedding_dim):
        """
        Initializes a linear embedding layer for each joint.
        """
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList([nn.Linear(joint_dim, embedding_dim) for _ in range(num_joints)])

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
        joint_embeddings = []
        for i in range(self.num_joints):
            j_i = src_seqs[:, :, i * self.joint_dim:(i+1) * self.joint_dim]
            joint_embeddings.append(self.embeddings[i](j_i))

        embedded_seqs = torch.cat(joint_embeddings, dim=-1)
        return embedded_seqs


class SpatioTemporalAttentionBlock(nn.Module):
    ...


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_joints, joint_dim, input_dim, embedding_dim):
        """
        """
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Define Modules
        print(embedding_dim * num_joints, embedding_dim * joint_dim)
        self.joint_embedder = JointEmbedder(num_joints, joint_dim, embedding_dim)
        self.output = nn.Linear(num_joints * embedding_dim, num_joints * joint_dim)  # TODO: Delete this placeholder

    def forward(self, src_seqs):
        """
        """
        embeddings_seqs = self.joint_embedder(src_seqs)
        print(src_seqs.shape, embeddings_seqs.shape)
        return self.output(embeddings_seqs)
