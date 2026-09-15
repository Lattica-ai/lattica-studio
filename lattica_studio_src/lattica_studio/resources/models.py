import os
from collections.abc import Iterable

from lattica_query.logging import (
    STUDIO_THEME,
    OperationLog,
    log_size_info,
)
from lattica_query.transport.backend import BackendAPI

from ..display import display_table
from ..exceptions import InvalidResourceResponseError, ResourceNotFoundError
from ..types import InstanceType, JsonDict, Model, ModelId


class ModelsAPI:
    def __init__(self, http: BackendAPI):
        self._http = http

    def create(
        self,
        name: str,
        instance_type: InstanceType,
        *,
        num_devices: int = 1,
    ) -> ModelId:
        """Create a model."""
        model_id = self._http.call(
            "api/model/create_model",
            parameters={
                "modelName": name,
                "instanceTypeId": instance_type.value,
                "numDevices": num_devices,
            },
        )

        return model_id

    def get_by_id(self, model_id: ModelId) -> Model:
        """Retrieve information about a model."""
        response = self._http.call(
            "api/model/get_model_info",
            parameters={
                "modelId": model_id,
            },
        )

        if not isinstance(response, dict):
            raise InvalidResourceResponseError("Model response is malformed")
        data = response.get("model")
        if not isinstance(data, dict):
            raise InvalidResourceResponseError("Model response does not contain a model object")
        return Model.from_api(data)

    def list(
        self,
        *,
        visibility: str | None = None,
    ) -> list[Model]:
        """List models."""
        params = {}

        if visibility is not None:
            params["visibility"] = visibility

        response = self._http.call(
            "api/model/list_models",
            parameters=params,
        )

        if not isinstance(response, dict):
            raise InvalidResourceResponseError("Model list response is malformed")
        models = response.get("models", [])
        if not isinstance(models, list) or not all(isinstance(model, dict) for model in models):
            raise InvalidResourceResponseError("Model list response is malformed")
        return [Model.from_api(model) for model in models]

    def find_by_name(
        self,
        name: str,
    ) -> Model | None:
        """Return a model with the given name, if one exists."""
        return next(
            (
                model
                for model in self.list()
                if model.name == name
            ),
            None,
        )

    def get_by_name(self, name: str) -> Model:
        """Retrieve a model by its exact name."""
        model = self.find_by_name(name)

        if model is None:
            raise ResourceNotFoundError(
                f"Model '{name}' does not exist."
            )

        return model

    @staticmethod
    def display(models: Iterable[Model]) -> None:
        """Print an easy-to-scan table of models."""
        display_table(
            ("NAME", "MODEL ID", "STATUS", "COMPILED", "INSTANCE"),
            (
                (
                    model.name,
                    model.id,
                    model.status,
                    "yes" if model.is_compiled else "no" if model.is_compiled is False else None,
                    model.instance_type,
                )
                for model in models
            ),
            empty_message="No models found.",
        )

    def update(
        self,
        model_id: ModelId,
        *,
        name: str | None = None,
        description: str | None = None,
        visibility: str | None = None,
        auto_restart: bool | None = None,
        input_type: str | None = None,
        output_type: str | None = None,
        status: str | None = None,
        instance_type: InstanceType | None = None,
    ) -> JsonDict:
        """Update model configuration."""
        params: JsonDict = {
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

        response = self._http.call(
            "api/model/update",
            parameters=params,
        )

        return {
            "message": response.get("message"),
            "modelId": response.get("modelId"),
            "warning": response.get("warning"),
        }

    def activate(self, model_id: ModelId) -> str:
        """Activate a model."""
        response = self._http.call(
            "api/model/activate_model",
            parameters={
                "modelId": model_id,
            },
        )

        return response["message"]

    def deactivate(self, model_id: ModelId) -> str:
        """Deactivate a model."""
        response = self._http.call(
            "api/model/deactivate_model",
            parameters={
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
        response = self._http.call(
            "api/model/update_model_visibility",
            parameters={
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
        with OperationLog(
            "uploading model",
            theme=STUDIO_THEME,
        ):
            log_size_info(
                "model",
                os.path.getsize(path),
            )

            self._http.upload_binary(
                "api/files/upload_non_homomorphic_model",
                parameters={
                    "modelId": model_id,
                },
                file_path=path,
            )
