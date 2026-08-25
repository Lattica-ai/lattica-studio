import os
from typing import Optional

from lattica_query.api.app import AppAPI
from lattica_studio.types import InstanceType
from lattica_query.logging import (
    Logging,
    STUDIO_THEME,
    log_size_info,
)

from ..types import JsonDict, ModelId, ModelInfo


class ModelsAPI:
    def __init__(self, http: AppAPI):
        self._http = http

    def create(
        self,
        name: str,
        instance_type: InstanceType,
        *,
        num_devices: int = 1,
    ) -> ModelId:
        """Create a model."""
        model_id = self._http.send_http_request(
            "api/model/create_model",
            req_params={
                "modelName": name,
                "instanceTypeId": instance_type.value,
                "numDevices": num_devices,
            },
        )

        return model_id

    def get(self, model_id: ModelId) -> ModelInfo:
        """Retrieve information about a model."""
        response = self._http.send_http_request(
            "api/model/get_model_info",
            req_params={
                "modelId": model_id,
            },
        )

        return response.get("model", {})

    def list(
        self,
        *,
        visibility: Optional[str] = None,
    ) -> list[ModelInfo]:
        """List models."""
        params = {}

        if visibility is not None:
            params["visibility"] = visibility

        response = self._http.send_http_request(
            "api/model/list_models",
            req_params=params,
        )

        return response.get("models", [])

    def find_by_name(
        self,
        name: str,
    ) -> Optional[ModelInfo]:
        """Return a model with the given name, if one exists."""
        return next(
            (
                model
                for model in self.list()
                if model.get("modelName") == name
            ),
            None,
        )

    def get_id_by_name(self, name: str) -> ModelId:
        model = self.find_by_name(name)

        if model is None:
            raise ValueError(
                f"Model '{name}' does not exist."
            )

        model_id = model.get("modelId")

        if model_id is None:
            raise RuntimeError(
                f"Model '{name}' does not contain a modelId."
            )

        return model_id

    def update(
        self,
        model_id: ModelId,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        visibility: Optional[str] = None,
        auto_restart: Optional[bool] = None,
        input_type: Optional[str] = None,
        output_type: Optional[str] = None,
        status: Optional[str] = None,
        instance_type: Optional[InstanceType] = None,
    ) -> JsonDict:
        """Update model configuration."""
        params = {
            "modelId": model_id,
        }

        if name is not None:
            params["modelName"] = name

        if description is not None:
            params["description"] = description

        if visibility is not None:
            params["visibility"] = visibility

        if auto_restart is not None:
            params["autoRestart"] = auto_restart

        if input_type is not None:
            params["inputType"] = input_type

        if output_type is not None:
            params["outputType"] = output_type

        if status is not None:
            params["status"] = status

        if instance_type is not None:
            params["instanceTypeId"] = instance_type.value

        response = self._http.send_http_request(
            "api/model/update",
            req_params=params,
        )

        return {
            "message": response.get("message"),
            "modelId": response.get("modelId"),
            "warning": response.get("warning"),
        }

    def activate(self, model_id: ModelId) -> str:
        """Activate a model."""
        response = self._http.send_http_request(
            "api/model/activate_model",
            req_params={
                "modelId": model_id,
            },
        )

        return response["message"]

    def deactivate(self, model_id: ModelId) -> str:
        """Deactivate a model."""
        response = self._http.send_http_request(
            "api/model/deactivate_model",
            req_params={
                "modelId": model_id,
            },
        )

        return response["message"]

    def set_visibility(
        self,
        model_id: ModelId,
        visibility: str,
    ) -> JsonDict:
        """Update a model's visibility."""
        response = self._http.send_http_request(
            "api/model/update_model_visibility",
            req_params={
                "modelId": model_id,
                "visibility": visibility,
            },
        )

        return {
            "message": response["message"],
            "modelId": response["modelId"],
            "newVisibility": response["newVisibility"],
        }

    def upload_plain(
        self,
        model_id: ModelId,
        path: str,
    ) -> None:
        """Upload a non-homomorphic model file."""
        with Logging(
            "uploading model",
            theme=STUDIO_THEME,
        ):
            log_size_info(
                "model",
                os.path.getsize(path),
            )

            self._http.send_http_file_request(
                "api/files/upload_non_homomorphic_model",
                req_params={
                    "modelId": model_id,
                },
                model_file_path=path,
            )