import torch


def generate_motion(model, src_seq, seq_len):
    """
    Function generates motion from a src_sequence of length seq_len.

    Args:
        model: A trained auto-regressive model
        src_seq: An input motion sequence
        seq_len: The length of the generated sequence

    Returns:
        A tensor of motion of shape (B, seq_len, D).
    """
    B, _, D = src_seq.shape
    print(f"\tGenerating sequences of shape: ({B}, {seq_len}, {D})")

    output_seqs = []
    output_seq = src_seq
    for _ in range(seq_len):
        output_seq = model(output_seq)
        output_seqs.append(output_seq[:, -1, :].unsqueeze(1))

    return torch.cat(output_seqs, dim=1)
