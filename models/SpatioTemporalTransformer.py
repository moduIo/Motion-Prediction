import torch


class SpatioTemporalTransformer(torch.nn.Module):
    def __init__(self, num_joints, joint_dim):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim

        self.W = torch.nn.Parameter(torch.randn((64, 120, 120)))

    def forward(self, src_seqs):
        """
        """
        print(type(src_seqs))
        return torch.bmm(self.W, src_seqs)

    def iterate_over_joints(self):
        # Iterate over joints
        for i in range(self.num_joints):
            j_i = self.src_seqs[0, 0, i*self.M:(i+1)*self.M]
