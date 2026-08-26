import unittest
from unittest import mock

from lattica_studio.resources import workers as workers_module
from lattica_studio.resources.workers import WorkersAPI
from lattica_studio.types import Worker


class WorkerDecisionLoggingTests(unittest.TestCase):
    def _workers(self):
        return WorkersAPI(mock.Mock())

    def test_get_or_start_logs_reused_worker_identifiers(self):
        workers = self._workers()
        ready = Worker(session_id="worker-1", model_id="model-1", status="UP")
        workers.active = mock.Mock(return_value=[ready])
        workers.start = mock.Mock()

        with mock.patch.object(workers_module, "log_info") as log_info:
            result = workers.get_or_start("model-1")

        self.assertIs(result, ready)
        workers.start.assert_not_called()
        messages = [call.args[0] for call in log_info.call_args_list]
        self.assertIn("model: model-1", messages)
        self.assertIn("decision: reuse ready worker", messages)
        self.assertIn("worker session: worker-1", messages)

    def test_get_or_start_logs_replacement_and_start(self):
        workers = self._workers()
        starting = Worker(session_id="worker-old", model_id="model-1", status="STARTING")
        replacement = Worker(session_id="worker-new", model_id="model-1", status="UP")
        workers.active = mock.Mock(return_value=[starting])
        workers.stop = mock.Mock()
        workers.start = mock.Mock(return_value=replacement)

        with mock.patch.object(workers_module, "log_info") as log_info:
            result = workers.get_or_start("model-1")

        self.assertIs(result, replacement)
        workers.stop.assert_called_once_with(model_id="model-1", session_id="worker-old")
        workers.start.assert_called_once_with("model-1")
        messages = [call.args[0] for call in log_info.call_args_list]
        self.assertIn("decision: replace non-ready worker", messages)
        self.assertIn("decision: no ready worker available; start a new worker", messages)

    def test_running_logs_and_honors_stop_on_exit(self):
        workers = self._workers()
        ready = Worker(session_id="worker-1", model_id="model-1", status="UP")
        workers.get_or_start = mock.Mock(return_value=ready)
        workers.stop = mock.Mock()

        with mock.patch.object(workers_module, "log_info") as log_info:
            with workers.running("model-1", stop_on_exit=True):
                pass

        workers.stop.assert_called_once_with(model_id="model-1", session_id="worker-1")
        messages = [call.args[0] for call in log_info.call_args_list]
        self.assertIn("stop on exit: True", messages)
        self.assertIn("decision: stop worker", messages)


if __name__ == "__main__":
    unittest.main()
