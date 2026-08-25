import unittest
from unittest import mock

from lattica_studio.deployment import DeploymentAPI
from lattica_studio.types import InstanceType, Model


class DeploymentKeyCacheInvalidationTests(unittest.TestCase):
    def _deployment(self):
        return DeploymentAPI(http=mock.Mock(), models=mock.Mock(), workers=mock.Mock())

    def test_redeploy_existing_model_invalidates_local_key_cache(self):
        deployment = self._deployment()
        existing_model = Model(
            id="model-1",
            name="MNIST_FC",
            instance_type=InstanceType.G6E_2XLARGE.value,
            num_devices=1,
        )
        deployment._models.find_by_name.return_value = existing_model

        with mock.patch("lattica_studio.deployment.invalidate_local_key_cache") as invalidate_cache:
            model_id = deployment._prepare_model(
                model_name="MNIST_FC",
                instance_type=InstanceType.G6E_2XLARGE,
                num_devices=1,
            )

        self.assertEqual(model_id, "model-1")
        deployment._workers.stop.assert_called_once_with(model_id="model-1")
        deployment._models.update.assert_not_called()
        invalidate_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

