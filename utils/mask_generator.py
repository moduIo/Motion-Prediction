import torch


class BatchJointMaskGenerator:
    def __init__(
        self,
        num_joints,
        joint_dim,
        mask_value=0,
        time_step_mask_prob=1.0,
        joint_mask_prob=1/24,
    ):
        """
        Args:
            num_joints: 15 or 24 for AMASS depending on how dataset is processed
            joint_dimension: 3 or 9 depending if aa or rotmat format is utilized
            mask_value: value to set masked joints as
            time_step_mask_prob: probability a time step will have at least one masked joint
            joint_mask_prob: probability that a joint will be masked within a time step
        """
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.mask_value = mask_value
        self.time_step_mask_prob = time_step_mask_prob
        self.joint_mask_prob = joint_mask_prob

        print(self.mask_value)

    def mask_joints(self, batch_sequence):
        """
        Masks joints in the batch sequence based on specified probabilities.

        Args:
            batch_sequence: A (batch_size, seq_len, dim) Tensor of sequences to process.

        Returns:
            A Tensor with some masked joints
        """
        B, T, _ = batch_sequence.shape
        masked_batch_sequence = batch_sequence.clone()

        for b in range(B):
            for t in range(T):
                if torch.rand(1).item() < self.time_step_mask_prob:
                    joints_to_mask = torch.rand(self.num_joints) < self.joint_mask_prob
                    indices_to_mask = torch.where(joints_to_mask)[0]
                    for n in indices_to_mask:
                        start_idx = n * self.joint_dim
                        end_idx = start_idx + self.joint_dim
                        masked_batch_sequence[b, t, start_idx:end_idx] = self.mask_value

        return masked_batch_sequence
