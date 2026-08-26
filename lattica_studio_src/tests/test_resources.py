import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from lattica_studio.exceptions import ResourceNotFoundError
from lattica_studio.resources.models import ModelsAPI
from lattica_studio.resources.tokens import TokensAPI
from lattica_studio.resources.workers import WorkersAPI
from lattica_studio.types import Model, TokenInfo, Worker


class ResourceObjectTests(unittest.TestCase):
    def test_models_are_attribute_objects_with_concise_repr(self):
        http = mock.Mock()
        http.send_http_request.return_value = {
            "models": [{
                "modelId": "12345678-1234-1234-1234-123456789abc",
                "modelName": "mnist",
                "status": "ACTIVE",
                "is_compiled": True,
                "requiredResources": {"gpuMemoryGB": [0.25]},
            }]
        }

        models = ModelsAPI(http).list()

        self.assertEqual(models[0].name, "mnist")
        self.assertTrue(models[0].is_compiled)
        self.assertEqual(models[0].required_resources, {"gpuMemoryGB": [0.25]})
        self.assertEqual(
            repr(models[0]),
            "Model(name='mnist', id='12345678…9abc', status='ACTIVE', compiled=True)",
        )

    def test_model_lookup_uses_typed_not_found_error(self):
        api = ModelsAPI(mock.Mock())
        api.list = mock.Mock(return_value=[])

        with self.assertRaisesRegex(ResourceNotFoundError, "missing"):
            api.get_id_by_name("missing")

    def test_active_workers_are_flat_attribute_objects(self):
        http = mock.Mock()
        http.send_http_request.return_value = {
            "activeWorkers": [{
                "workers": [{
                    "workerSessionId": "worker-1",
                    "modelId": "model-1",
                    "status": "UP",
                }]
            }]
        }

        with mock.patch("lattica_studio.resources.workers.log_info"):
            workers = WorkersAPI(http).active("model-1")

        self.assertEqual(workers, [Worker("worker-1", "model-1", "UP")])
        self.assertTrue(workers[0].is_ready)

    def test_token_list_returns_objects(self):
        http = mock.Mock()
        http.send_http_request.return_value = {
            "tokens": [{"tokenId": "token-1", "tokenName": "demo", "status": "ACTIVE"}]
        }

        tokens = TokensAPI(http).list()

        self.assertEqual(tokens, [TokenInfo(id="token-1", name="demo", status="ACTIVE")])

    def test_display_is_explicit_and_tabular(self):
        output = io.StringIO()
        models = [Model(id="model-1", name="mnist", status="ACTIVE", is_compiled=True)]

        with redirect_stdout(output):
            ModelsAPI.display(models)

        rendered = output.getvalue()
        self.assertIn("NAME", rendered)
        self.assertIn("MODEL ID", rendered)
        self.assertIn("mnist", rendered)
        self.assertIn("model-1", rendered)


if __name__ == "__main__":
    unittest.main()
