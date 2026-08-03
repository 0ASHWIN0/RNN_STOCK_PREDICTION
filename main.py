import argparse

from Rnn import BATCH_SIZE, EPOCHS, SEED, START, TICKER, main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an LSTM to forecast a stock's next-day high price."
    )
    parser.add_argument(
        "--ticker",
        default=TICKER,
        help="Stock ticker symbol (default: %(default)s)",
    )
    parser.add_argument(
        "--start",
        default=START,
        help="Start date in YYYY-MM-DD format (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Training batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
