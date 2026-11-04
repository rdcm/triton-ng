import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        self.target = config["parameters"]["target"]["string_value"]

    def execute(self, requests):
        responses = []
        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "Input3")
            stream = pb_utils.InferenceRequest(
                model_name=self.target,
                requested_output_names=["DIGIT"],
                inputs=[images],
            ).exec(decoupled=True)
            responses.append(answer(stream))
        return responses


def answer(stream):
    digits = []
    for result in stream:
        if result.has_error():
            return pb_utils.InferenceResponse(error=result.error())
        if len(result.output_tensors()) > 0:
            digit = pb_utils.get_output_tensor_by_name(result, "DIGIT").as_numpy()
            digits.extend(digit.flatten())
    digits = np.array(digits, dtype=np.int32)
    return pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("DIGIT", digits)])
