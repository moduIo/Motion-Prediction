from utils.dataset import prepare_dataset


def main():
    # Process train, val, test datasets
    dataset = prepare_dataset()

    # Setup Model
    _, seq_len, num_predictions = next(iter(dataset["train"]))[1].shape
    num_joints = 24  # AMASS DIP has 24 joints
    M = num_predictions // num_joints

    # TODO: Custom function for X, Y processing

    # Train
    for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
        print(src_seqs.shape, tgt_seqs.shape)

        for i in range(num_joints):
            j_i = src_seqs[0,0,i*M:(i+1)*M]
            print(j_i.shape)
            print(j_i)

        break


if __name__ == "__main__":
    main()
