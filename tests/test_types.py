from ray_unsloth.types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    GeneratedSequence,
    ImageChunk,
    ImmediateFuture,
    ModelInput,
    SampleResponse,
    SamplingParams,
    TensorData,
    to_plain_data,
)


def test_model_input_round_trips_ints():
    model_input = ModelInput.from_ints([1, 2, 3])

    assert model_input.to_ints() == [1, 2, 3]
    assert model_input.chunks == [EncodedTextChunk(tokens=[1, 2, 3])]
    assert model_input.length == 3
    assert ModelInput.empty().append(model_input).append_int(4).to_ints() == [1, 2, 3, 4]


def test_model_input_accepts_tinker_chunks_and_rejects_image_to_ints():
    model_input = ModelInput(chunks=[EncodedTextChunk(tokens=[1]), EncodedTextChunk(tokens=[2, 3])])

    assert model_input.to_ints() == [1, 2, 3]
    assert model_input.append(EncodedTextChunk(tokens=[4])).to_ints() == [1, 2, 3, 4]

    image_input = model_input.append(ImageChunk(data=b"jpeg", format="jpeg", expected_tokens=8))
    assert image_input.length == 11
    try:
        image_input.to_ints()
    except ValueError as exc:
        assert "EncodedTextChunks" in str(exc)
    else:
        raise AssertionError("expected image-containing ModelInput.to_ints() to fail")


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


def test_adam_params_accepts_tinker_field_names():
    params = AdamParams(
        learning_rate=1e-4,
        beta1=0.8,
        beta2=0.95,
        grad_clip_norm=1.0,
    )

    assert params.betas == (0.8, 0.95)
    assert params.beta1 == 0.8
    assert params.beta2 == 0.95
    assert params.max_grad_norm == 1.0
    assert params.grad_clip_norm == 1.0


def test_sample_response_has_tinker_compatible_fields():
    response = SampleResponse(
        sequences=[GeneratedSequence(tokens=[2], text="ok", stop_reason="length")],
        prompt_logprobs=[None, -0.1],
        topk_prompt_logprobs=[[], [(2, -0.1)]],
    )

    assert response.type == "sample"
    assert response.sequences[0].finish_reason == "length"
    assert response.sequences[0].stop_reason == "length"


def test_datum_converts_tensor_like_loss_inputs():
    class FakeTensor:
        dtype = "int64"
        shape = (2,)

        def tolist(self):
            return [1, 2]

    datum = Datum(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"labels": FakeTensor()},
    )

    assert datum.loss_fn_inputs["labels"] == TensorData(data=[1, 2], dtype="int64", shape=[2])


def test_tensor_data_from_array_helpers_round_trip():
    import numpy as np
    import torch

    from_numpy = TensorData.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.int64))
    from_torch = TensorData.from_torch(torch.tensor([0.25, 0.75], dtype=torch.float32))

    assert from_numpy.data == [1, 2, 3, 4]
    assert from_numpy.tolist() == [[1, 2], [3, 4]]
    assert from_torch.dtype == "float32"
    assert from_torch.tolist() == [0.25, 0.75]


def test_tinker_import_alias_exposes_types():
    import tinker
    from tinker.lib.public_interfaces import APIFuture
    from tinker.types.tensor_data import TensorData as TinkerTensorData

    assert tinker.ModelInput.from_ints([1]).to_ints() == [1]
    assert tinker.types.EncodedTextChunk(tokens=[1]).length == 1
    assert TinkerTensorData(data=[1], dtype="int64").tolist() == [1]
    assert APIFuture[tinker.ForwardBackwardOutput]
