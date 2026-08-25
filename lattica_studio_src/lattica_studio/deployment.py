import os
import tempfile
import time
from typing import TYPE_CHECKING

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.params.params import HomParams
from lattica_build import build, BuildArtifact
from lattica_query.api.app import AppAPI
from lattica_studio.types import InstanceType
from lattica_query.logging import (
    Logging,
    STUDIO_THEME,
    log_info,
    log_size_info,
    log_status,
)
from lattica_query.storage.tokens import invalidate_local_key_cache

from .exceptions import (
    CompilationError,
    CompilationTimeoutError,
    InvalidResourceResponseError,
)
from .resources.models import ModelsAPI
from .resources.workers import WorkersAPI
from .types import ModelId


class DeploymentAPI:
    def __init__(
        self,
        *,
        http: AppAPI,
        models: ModelsAPI,
        workers: WorkersAPI,
    ):
        self._http = http
        self._models = models
        self._workers = workers

    def deploy(
            self,
            artifact: BuildArtifact,
            model_name: str,
            instance_type: InstanceType = InstanceType.G7E_2XLARGE,
            num_devices: int = 1,
    ) -> ModelId:

        model_id = self._prepare_model(
            model_name=model_name,
            instance_type=instance_type,
            num_devices=num_devices,
        )

        self._upload_and_compile(
            path=artifact.path,
            model_id=model_id,
            init_context_params=artifact.init_context_params,
        )

        return model_id

    def deploy_pipeline(
        self,
        hom_pipeline: "HomomorphicPipeline",
        hom_params: "HomParams",
        model_name: str,
        instance_type: InstanceType = InstanceType.G6E_2XLARGE,
        num_devices: int = 1,
        display_graph: bool = False,
    ) -> ModelId:
        """
        Deploy a homomorphic pipeline.

        If a model with the same name already exists, redeploy into that
        model instead of creating a duplicate.
        """
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp_file:
            artifact = build(
                hom_pipeline,
                hom_params,
                tmp_file.name,
                display_graph=display_graph
            )

            return self.deploy(
                artifact,
                model_name,
                instance_type=instance_type,
                num_devices=num_devices,
            )

    def _prepare_model(
        self,
        *,
        model_name: str,
        instance_type: InstanceType,
        num_devices: int,
    ) -> ModelId:
        with Logging(
            "registering model",
            theme=STUDIO_THEME,
        ):
            existing_model = self._models.find_by_name(
                model_name
            )

            if existing_model is None:
                log_info(
                    f"model '{model_name}' doesn't exist; "
                    "creating a new model"
                )

                model_id = self._models.create(
                    model_name,
                    instance_type,
                    num_devices=num_devices,
                )

                log_info(
                    f"model ID: {model_id}"
                )

                return model_id

            model_id = existing_model.id
            if model_id is None:
                raise InvalidResourceResponseError(
                    f"Model '{model_name}' does not contain a model ID"
                )

            log_info(
                f"model '{model_name}' already exists; "
                f"redeploying into model {model_id}"
            )

            existing_num_devices = existing_model.num_devices or 1

            if existing_num_devices != num_devices:
                raise ValueError(
                    f"Model '{model_name}' was created with "
                    f"num_devices={existing_num_devices}, but "
                    f"num_devices={num_devices} was requested. "
                    "Deploy under a different model name."
                )

            log_info(
                f"stopping active workers for model '{model_name}'"
            )

            self._workers.stop(
                model_id=model_id,
            )

            current_instance_type = existing_model.instance_type

            if current_instance_type != instance_type.value:
                log_info(
                    f"updating instance type to "
                    f"{instance_type.value}"
                )

                self._models.update(
                    model_id,
                    instance_type=instance_type,
                )

            # Redeploy may change preprocessing/model metadata; drop stale local key bundle.
            invalidate_local_key_cache()

            return model_id

    def _upload_and_compile(
        self,
        *,
        path: str,
        model_id: ModelId,
        init_context_params: dict,
    ) -> None:
        with Logging(
            "uploading model",
            theme=STUDIO_THEME,
        ):
            log_size_info(
                "model",
                os.path.getsize(path),
            )

            self._http.upload_file_and_alert(
                path,
                endpoint="api/model/get_model_upload_url",
                upload_params={
                    "modelId": model_id,
                },
                alert_params=init_context_params,
            )

            log_status(
                "registering upload"
            )

        self._wait_for_compilation(
            model_id
        )

    def _wait_for_compilation(
        self,
        model_id: ModelId,
        *,
        poll_interval: float = 5,
        timeout: float = 600,
    ) -> None:
        start_time = time.monotonic()

        with Logging(
            "compiling model",
            theme=STUDIO_THEME,
        ):
            while True:
                if time.monotonic() - start_time >= timeout:
                    raise CompilationTimeoutError(
                        f"Model {model_id} compilation timed out "
                        f"after {timeout:g} seconds."
                    )

                log_status(
                    "checking compilation status"
                )

                model = self._models.get(
                    model_id
                )

                if model.is_compiled:
                    log_status(
                        "compilation complete"
                    )

                    log_info(
                        f"model ID: {model_id}"
                    )

                    return

                if model.status == "INACTIVE":
                    message = (
                        f"Model {model_id} compilation failed "
                        "(model status: INACTIVE)"
                    )

                    compilation_error = model.compilation_error

                    if compilation_error:
                        message += (
                            f": {compilation_error}"
                        )

                    raise CompilationError(
                        message
                    )

                status = model.status

                if status:
                    log_status(
                        f"compiling • {status.lower()}"
                    )
                else:
                    log_status(
                        "compiling"
                    )

                time.sleep(
                    poll_interval
                )
