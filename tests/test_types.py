from ray_unsloth.types import Datum, ImmediateFuture, ModelInput, SamplingParams, to_plain_data


def test_model_input_round_trips_ints():
    model_input = ModelInput.from_ints([1, 2, 3])

    assert model_input.to_ints() == [1, 2, 3]


def test_immediate_future_matches_ray_future_shape():
    future = ImmediateFuture({"ok": True})

    assert future.result() == {"ok": True}
    assert future.get() == {"ok": True}


def test_to_plain_data_handles_nested_dataclasses():
    datum = Datum(
        model_input=ModelInput.from_ints([1]),
        loss_fn_inputs={"params": SamplingParams(max_tokens=4)},
    )

    assert to_plain_data(datum)["model_input"]["tokens"] == [1]
    assert to_plain_data(datum)["loss_fn_inputs"]["params"]["max_tokens"] == 4
