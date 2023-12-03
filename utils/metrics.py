from matplotlib import pyplot as plt


def plot_training_metrics(args, training_losses, validation_losses):
    """
    Function plots the training/val metrics

    Args:
        args: The args for the run
        training_losses: A list of the training losses
        validation_losses: A list of the validation losses

    Returns:
        None
    """
    plt.plot(range(len(training_losses)), training_losses)
    plt.plot(range(len(validation_losses)), validation_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"{args.save_model_path}/loss.svg", format="svg")
    plt.savefig(f"{args.save_model_path}/loss.png", format="png")
