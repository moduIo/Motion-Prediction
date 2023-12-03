def generate_motion(model, src_seq, seq_len):
    print(f"\tGenerating sequence of length: {seq_len}")
    return src_seq[:, :seq_len, :]
