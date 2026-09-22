from abc import ABC, abstractmethod

import torch

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.params.params import HomParams


DEFAULT_ACCURACY = 1 / 2 ** 10


class PipelineWrapper(ABC):
    """User-facing definition of a buildable Lattica pipeline."""

    hom_pipeline = None
    exmpl_pt = None
    custom_data = None
    serialized_data = b''

    @abstractmethod
    def build_pipeline(self) -> HomomorphicPipeline:
        raise NotImplementedError

    @abstractmethod
    def build_params(self) -> HomParams:
        raise NotImplementedError

    def get_hom_pipeline(self) -> HomomorphicPipeline:
        if self.hom_pipeline is None:
            self.hom_pipeline = self.build_pipeline()
        return self.hom_pipeline

    def get_hom_params(self) -> HomParams:
        return self.build_params()

    def _set_example_pt(self) -> torch.Tensor:
        pipeline = self.get_hom_pipeline()
        return torch.rand(pipeline.input_shape[pipeline.primary_input_name])

    def get_example_pt(self) -> torch.Tensor:
        self.exmpl_pt = self._set_example_pt()
        return self.exmpl_pt

    def _set_custom_data(self) -> None:
        self.custom_data = {}

    def get_custom_data(self) -> dict:
        self._set_custom_data()
        return self.custom_data

    def set_preprocessing_data(self, preprocessing_data: bytes) -> None:
        self.serialized_data = preprocessing_data

    def compute_expected(self, example_pt: torch.Tensor) -> torch.Tensor:
        """Compute the true expected result for an example input.

        Examples should override this when they provide a clear reference
        implementation. Some pipelines intentionally do not provide one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define compute_expected()"
        )

    def display_results(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        print(f'{actual=}')
        print(f'{expected=}')
        print(f'{actual.shape=} {expected.shape=}')

    def verify_results(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=DEFAULT_ACCURACY,
            atol=DEFAULT_ACCURACY,
        )
