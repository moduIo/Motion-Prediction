def generate_motion(model, src_seq, seq_len):
    return src_seq[:, :seq_len, :]
