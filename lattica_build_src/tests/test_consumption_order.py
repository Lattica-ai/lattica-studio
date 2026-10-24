"""The level tracing records the order a pipeline consumes its rows, row budgets included, on the parameters."""
import numpy as np

from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.operators.polynomials.h_poly_eval import HomPolyEval
from lattica_build.params.params import HomParams


def _params():
    return HomParams(n=256, full_q_list_precision=((60,),) * 5, pt_scale=2 ** 30, dnum=2, n_slots=8)


def test_serialize_records_the_consumption_order_with_row_budgets():
    # the first polynomial may only rescale with rows 3 and 4, the second takes the rest from the top
    pipeline = HomomorphicPipeline(
        hom=SequentialHomOp(HomPolyEval(func=np.abs, degree=3, rows_budget=[3, 4]),
                            HomPolyEval(func=np.abs, degree=3)),
        input_shape=(8,))
    params = _params()
    assert params.consumption_order is None
    pipeline.serialize(params)
    assert params.consumption_order[:2] == (3, 4)
    assert set(params.consumption_order) == {0, 1, 2, 3, 4}
    assert params.mod_chain.get_consumption_rows(params) == list(params.consumption_order)


def test_consumption_order_round_trips_and_defaults_to_index_order():
    d = _params().to_dict()
    assert d["consumption_order"] is None
    d["consumption_order"] = [4, 3, 2, 1, 0]
    restored = HomParams.from_dict(d)
    assert restored.consumption_order == (4, 3, 2, 1, 0)
