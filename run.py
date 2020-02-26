import argparse
from pathlib import Path
import sys
# pylint: disable=wrong-import-position
sys.path.append("landcover")
from train_model_landcover import Train
from testing_model_landcover import Test
from compute_accuracy import compute_accuracy
from eval_landcover_results import eval_landcover_results

from helpers import get_logger

logger = get_logger(__name__)


def do_args():
    parser = argparse.ArgumentParser(
        description="Wrapper utility for training and testing land cover models."
    )
    parser.add_argument(
        "-v", "--verbose", type=int, help="Verbosity of keras.fit", default=2
    )
    parser.add_argument("--name", type=str, help="Experiment name", required=True)
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to data directory containing the splits CSV files",
        required=True,
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output base directory", required=True,
    )
    parser.add_argument(
        "--training-states",
        nargs="+",
        type=str,
        help="States to use as training",
        required=True,
    )
    parser.add_argument(
        "--validation-states",
        nargs="+",
        type=str,
        help="States to use as validation",
        required=True,
    )
    parser.add_argument(
        "--superres-states",
        nargs="+",
        type=str,
        help="States to use only superres loss with",
        default=[],
    ),
    parser.add_argument(
        "--test-states",
        nargs="+",
        type=str,
        help="States to test model with",
        default=[],
    ),
    parser.add_argument(
        "--do-color",
        action="store_true",
        help="Enable color augmentation",
        default=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "unet_large", "fcdensenet", "fcn_small"],
        help="Model architecture to use",
        required=True,
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=100)
    parser.add_argument(
        "--loss",
        type=str,
        help="Loss function",
        default="crossentropy",
        choices=["crossentropy", "jaccard", "superres"],
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Learning rate", default=0.001,
    )
    parser.add_argument("--batch-size", type=int, help="Batch size", default=128)
    return parser.parse_args()


def main():
    # Read arguments
    args = do_args()
    logger.info(args)

    # Ensure folders are there and no overwrite
    logger.info("Ensuring all folders are there...")
    assert Path(args.data_dir).is_dir()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    assert Path(args.output_dir).is_dir()
    assert not (Path(args.output_dir) / Path(args.name)).is_dir()

    # Run training
    train = Train(
        name=args.name,
        output=args.output_dir,
        data_dir=args.data_dir,
        training_states=args.training_states,
        validation_states=args.validation_states,
        superres_states=args.superres_states,
        model_type=args.model,
        loss=args.loss,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        do_color=args.do_color,
        batch_size=args.batch_size,
    )
    train.run_experiment()

    for test_state in args.test_states:
        # Run testing
        ## Get test file name
        input_fn = Path(args.data_dir) / ("%s_extended-test_tiles.csv" % test_state)

        ## Get model file name
        model_fn = Path(args.output_dir) / args.name / "final_model.h5"

        prediction_dir = (
            Path(args.output_dir) / args.name / ("test-output_%s" % test_state)
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)

        test = Test(
            input_fn=input_fn,
            output_base=prediction_dir,
            model_fn=model_fn,
            save_probabilities=False,
            superres=args.loss == "superres",
        )
        test.run_on_tiles()

        # Run accuracy
        compute_accuracy(pred_dir=prediction_dir, input_fn=input_fn)

        # Run eval


if __name__ == "__main__":
    main()
