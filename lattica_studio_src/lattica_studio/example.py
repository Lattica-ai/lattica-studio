"""Build, deploy, and query an encrypted MNIST model end to end."""

import os
import struct
from pathlib import Path

import torch
from dotenv import load_dotenv
from lattica_build import build
from lattica_build.examples.advanced import mnist_fc
from lattica_query import OutputConfig, QueryClient, output_context

from lattica_studio import LatticaStudio, InstanceType

MODEL_NAME = "MNIST_FC_WINDOWS_CPP_G4_v2"
ARTIFACT_PATH = "mnist_fc_pipeline.zip"
NUM_QUERIES = 3
MIN_ACCURACY = 0.95
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_mnist_test_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Load the packaged IDX test set in pipeline-sized batches."""
    print("Loading MNIST test data...")
    data_path = Path(mnist_fc.__file__).with_name("data")
    images_path = data_path / "t10k-images-idx3-ubyte"
    labels_path = data_path / "t10k-labels-idx1-ubyte"
    image_data = images_path.read_bytes()
    label_data = labels_path.read_bytes()
    image_magic, image_count, rows, columns = struct.unpack(">IIII", image_data[:16])
    images = torch.frombuffer(bytearray(image_data[16:]), dtype=torch.uint8).reshape(
        image_count, rows, columns
    ).to(torch.float32)
    labels = torch.frombuffer(bytearray(label_data[8:]), dtype=torch.uint8).to(torch.int64)
    usable_count = (images.shape[0] // mnist_fc.BATCH) * mnist_fc.BATCH
    x = images[:usable_count].reshape((-1, *mnist_fc.INPUT_SHAPE))
    y = labels[:usable_count].reshape((-1, mnist_fc.BATCH))
    return x, y


def main() -> None:
    load_dotenv(ENV_PATH)
    license_key = os.getenv("LATTICA_LICENSE_KEY", "")
    if not license_key:
        raise ValueError("Set LATTICA_LICENSE_KEY to run this example")

    x, y = load_mnist_test_data()

    # Build the pipeline locally, then deploy and compile it.
    pipeline_definition = mnist_fc.Pipeline()
    hom_pipeline = pipeline_definition.build_pipeline()
    artifact = build(
        hom_pipeline,
        pipeline_definition.build_params(),
        ARTIFACT_PATH,
        display_graph=True,
    )

    # Optional, forward_clear runs the pipeline locally on plaintext tensors for verification.
    clear_result = hom_pipeline.forward_clear(x[0])
    clear_prediction = clear_result.argmax(dim=-1)
    clear_accuracy = (clear_prediction == y[0]).sum().item() / mnist_fc.BATCH
    print(f"Clear query: accuracy {clear_accuracy * 100:.1f}%")
    if clear_accuracy < MIN_ACCURACY:
        raise RuntimeError(
            f"Clear MNIST accuracy {clear_accuracy:.1%} is below "
            f"the required {MIN_ACCURACY:.1%}"
        )


    with LatticaStudio(license_key) as studio:
        # Optional, display the list of all models in the account.
        # models = studio.models.list()
        # studio.models.display(models)
        # Optional, stop all workers of all models in the account.
        # for model in models:
        #     studio.workers.stop(model.id)
        # Optional, deactivate all models in the account.
        # for model in models:
        #     studio.models.deactivate(model.id)

        model_id = studio.deploy(
            artifact,
            MODEL_NAME,
            instance_type=InstanceType(os.getenv("LATTICA_INSTANCE_TYPE", InstanceType.G4DN_XLARGE.value)),
        )
        # Optional, load existing model by name instead of deploying a new one
        model = studio.models.get_by_name(MODEL_NAME)
        model_id = model.id

        # A worker must be running to serve encrypted queries.
        with studio.workers.running(model_id, stop_on_exit=True):
            token = studio.tokens.create(model_id, name=MODEL_NAME, save=True)

            with QueryClient(token) as client:
                # Generates FHE keys and uploads the evaluation key.
                # The secret key never leaves this machine.
                sk = client.keys.ensure()

                for i in range(NUM_QUERIES):
                    print(f"Running encrypted query {i + 1}...")
                    print(f"{x[i].shape=}...")

                    result = client.query.encrypted(x[i], key=sk)

                    prediction = result.argmax(dim=-1)
                    accuracy = (prediction == y[i]).sum().item() / mnist_fc.BATCH

                    print(f"Query {i + 1}: accuracy {accuracy * 100:.1f}%")
                    if accuracy < MIN_ACCURACY:
                        raise RuntimeError(
                            f"MNIST query accuracy {accuracy:.1%} is below "
                            f"the required {MIN_ACCURACY:.1%}"
                        )


if __name__ == "__main__":
    with output_context(OutputConfig(color=True)):
        main()
