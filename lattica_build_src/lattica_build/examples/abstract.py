from abc import ABC
from typing import Optional, Tuple
import torch
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline

DEFAULT_ACCURACY = 1 / 2 ** 10

class Example(ABC):
    hom_pipeline = None
    exmpl_pt     = None

    def _set_hom_params(self):
        raise NotImplementedError

    def get_hom_params(self):
        return self._set_hom_params()

    def _set_hom_pipeline(self) -> 'HomomorphicPipeline':
        raise NotImplementedError

    def get_hom_pipeline(self) -> 'HomomorphicPipeline':
        if self.hom_pipeline is None:
            self.hom_pipeline = self._set_hom_pipeline()
        return self.hom_pipeline

    def _set_example_pt(self) -> 'Tensor':
        return torch.rand(self.hom_pipeline.input_shape[self.hom_pipeline.primary_input_name])

    def get_example_pt(self) -> 'Tensor':
        self.exmpl_pt = self._set_example_pt()
        return self.exmpl_pt

    def display_results(self, actual: 'Tensor', expected: 'Tensor') -> None:
        print(f'{actual=}')
        print(f'{expected=}')
        print(f'{actual.shape=} {expected.shape=}')

    def verify_results(self, actual: 'Tensor', expected: 'Tensor') -> None:
        torch.testing.assert_close(
            actual, expected,
            rtol=DEFAULT_ACCURACY, atol=DEFAULT_ACCURACY
        )
