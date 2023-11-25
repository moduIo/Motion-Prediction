import torch


class SpatioTemporalTransformer(torch.nn.Module):
    def __init__(self, num_joints, joint_dim, input_dim, embedding_dim):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.W = torch.nn.Parameter(torch.randn((input_dim, 256)))
        self.output = torch.nn.Parameter(torch.randn((256, input_dim)))

    def forward(self, src_seqs):
        """
        """
        print(self.W.shape, src_seqs.shape)
        # src_seqs.shape = torch.Size([64, 120, 72]) = [64, 120, num_joints * joint_dim]
        # 1. map each joint to it's embedding: [64, 120, 72] => [64, 120, num_joints * embedding_dim]
        # 2. Positional embedding
        return src_seqs @ self.W @ self.output

    def iterate_over_joints(self):
        # Iterate over joints
        for i in range(self.num_joints):
            j_i = self.src_seqs[0, 0, i*self.M:(i+1)*self.M]
