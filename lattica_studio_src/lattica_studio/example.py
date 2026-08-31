"""Build, deploy, and query an encrypted MNIST model end to end."""

import os

import mnist
import torch

from lattica_build import build
from lattica_build.examples.advanced import mnist_fc
from lattica_query import QueryClient
from lattica_studio import LatticaStudio


MODEL_NAME = "MNIST_FC"
ARTIFACT_PATH = "mnist_fc_pipeline.zip"
NUM_QUERIES = 3


def load_mnist_test_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Load MNIST test data, normalized and batched for the pipeline input."""
    print("Loading MNIST test data...")
    mnist.datasets_url = "https://raw.githubusercontent.com/fgnt/mnist/master/"

    images = mnist.test_images()
    labels = mnist.test_labels()

    x = torch.tensor(images, dtype=torch.float32).reshape((-1, *mnist_fc.INPUT_SHAPE))
    x = ((x / 255.0) - 0.1307) / 0.3081
    y = torch.tensor(labels, dtype=torch.long).reshape((-1, mnist_fc.BATCH))

    return x, y


def main() -> None:
    license_key = os.getenv("LATTICA_LICENSE_KEY", "")
    if not license_key:
        raise ValueError("Set LATTICA_LICENSE_KEY to run this example")

    x, y = load_mnist_test_data()

    studio = LatticaStudio(license_key)

    # Build the pipeline locally, then deploy and compile it.
    artifact = build(
        mnist_fc.build_pipeline(),
        mnist_fc.build_params(),
        ARTIFACT_PATH,
        display_graph=True,
    )

    # Optional, display the list of all models in the account.
    models = studio.models.list()
    # studio.models.display(models)
    # Optional, stop all workers of all models in the account.
    # for model in models:
    #     studio.workers.stop(model.id)
    # Optional, deactivate all models in the account.
    # for model in models:
    #     studio.models.deactivate(model.id)

    model_id = studio.deploy(artifact, MODEL_NAME)

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


if __name__ == "__main__":
    main()
