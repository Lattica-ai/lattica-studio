"""Build, deploy, and query an encrypted MNIST model end to end."""

import os
from pathlib import Path

import torch

from lattica_build import build
from lattica_build.examples.advanced import mnist_fc
from lattica_query import QueryClient
from lattica_studio import LatticaStudio

MODEL_NAME = "MNIST_FC"
ARTIFACT_PATH = "mnist_fc_pipeline.zip"
NUM_QUERIES = 3
MIN_ACCURACY = 0.95


def load_mnist_test_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Load the packaged raw MNIST examples in pipeline-sized batches."""
    print("Loading MNIST test data...")
    data_path = Path(mnist_fc.__file__).with_name("data") / "mnist_test_batch.pt"
    batch = torch.load(data_path, weights_only=True, map_location="cpu")
    images = batch["images"]
    labels = batch["labels"]

    usable_count = (images.shape[0] // mnist_fc.BATCH) * mnist_fc.BATCH
    x = images[:usable_count].reshape((-1, *mnist_fc.INPUT_SHAPE))
    y = labels[:usable_count].reshape((-1, mnist_fc.BATCH))

    return x, y


def main() -> None:
    license_key = os.getenv("LATTICA_LICENSE_KEY", "")
    if not license_key:
        raise ValueError("Set LATTICA_LICENSE_KEY to run this example")

    x, y = load_mnist_test_data()

    studio = LatticaStudio(license_key)

    # Build the pipeline locally, then deploy and compile it.
    pipeline = mnist_fc.build_pipeline()
    artifact = build(
        pipeline,
        mnist_fc.build_params(),
        ARTIFACT_PATH,
        display_graph=True,
    )

    # Optional, forward_clear runs the pipeline locally on plaintext tensors for verification.
    clear_result = pipeline.forward_clear(x[0])
    clear_prediction = clear_result.argmax(dim=-1)
    clear_accuracy = (clear_prediction == y[0]).sum().item() / mnist_fc.BATCH
    print(f"Clear query: accuracy {clear_accuracy * 100:.1f}%")
    if clear_accuracy < MIN_ACCURACY:
        raise RuntimeError(
            f"Clear MNIST accuracy {clear_accuracy:.1%} is below "
            f"the required {MIN_ACCURACY:.1%}"
        )

    # Optional, display the list of all models in the account.
    # models = studio.models.list()
    # studio.models.display(models)
    # Optional, stop all workers of all models in the account.
    # for model in models:
    #     studio.workers.stop(model.id)
    # Optional, deactivate all models in the account.
    # for model in models:
    #     studio.models.deactivate(model.id)

    model_id = studio.deploy(artifact, MODEL_NAME)

    # Optional, load existing model by name instead of deploying a new one
    # model_id = studio.models.get_id_by_name(MODEL_NAME)

    # A worker must be running to serve encrypted queries.
    with studio.workers.running(model_id, stop_on_exit=True):
        token = studio.tokens.create(model_id, save_as=MODEL_NAME)

        client = QueryClient(token)

        # Generates FHE keys and uploads the evaluation key.
        # The secret key never leaves this machine.
        sk = client.generate_key()

        for i in range(NUM_QUERIES):
            print(f"Running encrypted query {i + 1}...")
            print(f"{x[i].shape=}...")

            result = client.run_query(sk, x[i])

            prediction = result.argmax(dim=-1)
            accuracy = (prediction == y[i]).sum().item() / mnist_fc.BATCH

            print(f"Query {i + 1}: accuracy {accuracy * 100:.1f}%")
            if accuracy < MIN_ACCURACY:
                raise RuntimeError(
                    f"MNIST query accuracy {accuracy:.1%} is below "
                    f"the required {MIN_ACCURACY:.1%}"
                )


if __name__ == "__main__":
    main()
