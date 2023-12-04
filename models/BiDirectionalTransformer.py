import torch.nn as nn
from models.SpatioTemporalTransformer import JointEmbedding


class BiDirectionalTransformer(nn.Module):
    ### See how facebook implemented theirs in fair motion, add blocks for num heads, etc.
    def __init__(
        self,
        num_joints,
        joint_dim,
        input_dim,
        embedding_dim=32,
        nhead=12,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        batch_first=True,
    ):
        super().__init__()

        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.input_dim = input_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first
        self.embedding_dim = embedding_dim

        # Define Modules
        self.joint_embedder = JointEmbedding(
            self.num_joints, self.joint_dim, self.embedding_dim, self.dropout
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim * self.num_joints,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=self.batch_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )
        self.output_layer = nn.Linear(
            self.embedding_dim * self.num_joints, self.num_joints * self.joint_dim
        )

    def forward(self, src):
        """
        TODO: Method applies the linear joint embedding to each joint and concats them together.
        This maps from:
            (B, T, N*M) => (B, T, N*M)

        Where:
            1. B := Batch Size
            2. T := Sequence Length
            3. N := Number of Joints
            4. M := Joint Dimensions

        Args:
            src_seqs: A source Tensor of sequences (batch_size, seq_length, dim).

        Returns:
            output: A forward pass through the bidirectional transforemer.
        """

        if src.shape[2] != self.num_joints * self.joint_dim:
            raise ValueError("Invalid data type: expected (B, T, N*M) format")

        output = self.joint_embedder(src)
        output = self.transformer_encoder(output)
        output = self.output_layer(output)

        return output
