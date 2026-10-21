import unittest
from unittest import mock

from lattica_studio.deployment import DeploymentAPI
from lattica_studio.types import InstanceType, Model


class DeploymentPreparationTests(unittest.TestCase):
    def _deployment(self):
        return DeploymentAPI(http=mock.Mock(), models=mock.Mock(), workers=mock.Mock())

    def test_redeploy_existing_model_stops_workers(self):
        deployment = self._deployment()
        existing_model = Model(
            id="model-1",
            name="MNIST_FC",
            instance_type=InstanceType.G4DN_XLARGE.value,
            num_devices=1,
        )
        deployment._models.find_by_name.return_value = existing_model

        model_id = deployment._prepare_model(
            model_name="MNIST_FC",
            instance_type=InstanceType.G4DN_XLARGE,
            num_devices=1,
        )

        self.assertEqual(model_id, "model-1")
        deployment._workers.stop.assert_called_once_with(model_id="model-1")
        deployment._models.update.assert_not_called()

    def _deploy_new_model(self, **kwargs):
        deployment = self._deployment()
        deployment._models.find_by_name.return_value = None
        deployment._models.create.return_value = "model-1"
        with mock.patch.object(deployment, "_upload_and_compile"):
            deployment.deploy(mock.Mock(), "MNIST_FC", **kwargs)
        return deployment._models.create

    def test_single_device_defaults_to_g4dn(self):
        create = self._deploy_new_model()
        create.assert_called_once_with(
            "MNIST_FC", InstanceType.G4DN_XLARGE, num_devices=1
        )

    def test_multi_device_defaults_to_g7e_12xlarge(self):
        create = self._deploy_new_model(num_devices=2)
        create.assert_called_once_with(
            "MNIST_FC", InstanceType.G7E_12XLARGE, num_devices=2
        )

    def test_explicit_instance_type_is_kept(self):
        create = self._deploy_new_model(instance_type=InstanceType.G7E_12XLARGE)
        create.assert_called_once_with(
            "MNIST_FC", InstanceType.G7E_12XLARGE, num_devices=1
        )


if __name__ == "__main__":
    unittest.main()
