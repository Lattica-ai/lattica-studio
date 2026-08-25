"""Staged workflow for deploying and querying a model.

Stages:
- Stage A: deploy + compile only.
- Stage B: create query token, start worker, generate keys, register EK, stop worker.
- Stage C: recurring query runs using the already registered key context.
"""

from __future__ import annotations

import os
from asyncio import sleep
from typing import Any
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from lattica_build.examples import example_mnist_fc
from lattica_studio import LatticaStudio
from lattica_query import QueryClient


# Set explicitly, or keep empty to read LATTICA_LICENSE_KEY from environment.
LICENSE_KEY = ""
if not LICENSE_KEY:
    LICENSE_KEY = os.getenv("LATTICA_LICENSE_KEY", "")
if not LICENSE_KEY:
    raise ValueError("Set LICENSE_KEY or LATTICA_LICENSE_KEY before running this script")
MODEL_NAME = "MY_SHARPEN_MODEL"

# Run stages selectively during development.
RUN_DEPLOY_AND_COMPILE                   = True
RUN_CREATE_QUERY_TOKEN_AND_GENERATE_KEYS = True
RUN_ENCRYPTED_QUERY                      = True


# Load MNIST test data for a single batch to query the model.
print(f'Loading MNIST test data for a single batch to query the model...')
test_dataset = datasets.MNIST(
    "../data", train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
)
loader = DataLoader(test_dataset, batch_size=example_mnist_fc.BATCH, shuffle=True)
input_data = iter(loader)



def main() -> None:

    studio = LatticaStudio(LICENSE_KEY)

    if RUN_DEPLOY_AND_COMPILE:
        model_id = studio.deploy_pipeline(
            example_mnist_fc.build_pipeline(),
            example_mnist_fc.build_params(),
            MODEL_NAME,
            display_graph=True,
        )
    else:
        model_id = studio.models.get_id_by_name(MODEL_NAME)


    if RUN_CREATE_QUERY_TOKEN_AND_GENERATE_KEYS:
        token = studio.tokens.create(model_id, save_as=MODEL_NAME)
        with studio.workers.running(model_id, stop_on_exit=not RUN_ENCRYPTED_QUERY):
            query_client = QueryClient(token)
            query_client.generate_key(load_if_exists=False)
    else:
        token = studio.tokens.load(MODEL_NAME)
        query_client = QueryClient(token)


    if RUN_ENCRYPTED_QUERY:
        with studio.workers.running(model_id, stop_on_exit=True):
            sk = query_client.generate_key(load_if_exists=True)

            for _ in range(3):
                pt, ground_truth = next(input_data)
                # Run the query and print accuracy.
                res = query_client.run_query(sk, pt)
                y_pred = res.argmax(dim=-1)
                print(f"Accuracy: {(y_pred == ground_truth).sum().item() / example_mnist_fc.BATCH * 100:.1f}%")



if __name__ == "__main__":
    main()

