import torch


def generate_motion(model, src_seq, seq_len, save_path, save):
    """
    Function generates motion from a src_sequence of length seq_len.

    Args:
        model: A trained auto-regressive model
        src_seq: An input motion sequence
        seq_len: The length of the generated sequence
        save_path: The path to save the sequence to
        save: True if the motion sequence should be saved or not

    Returns:
        A tensor of motion of shape (B, seq_len, D).
    """
    B, _, D = src_seq.shape
    print(f"\tGenerating sequences of shape: ({B}, {seq_len}, {D})")

    # Generate output in a sliding window fashion
    output_seqs = []
    output_seq = src_seq
    for _ in range(seq_len):
        output_seq = model(output_seq)
        output_seqs.append(output_seq[:, -1, :].unsqueeze(1))

    output_seq = torch.cat(output_seqs, dim=1)

    if save:
        torch.save(output_seq, save_path)
    return output_seq
