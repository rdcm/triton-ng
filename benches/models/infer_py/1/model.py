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
            image = pb_utils.get_input_tensor_by_name(request, "Input3")
            result = pb_utils.InferenceRequest(
                model_name=self.target,
                requested_output_names=["Plus214_Output_0"],
                inputs=[image],
            ).exec()
            responses.append(answer(result))
        return responses


def answer(result):
    if result.has_error():
        return pb_utils.InferenceResponse(error=result.error())
    logits = pb_utils.get_output_tensor_by_name(result, "Plus214_Output_0").as_numpy()
    digit = np.array([[int(np.argmax(logits))]], dtype=np.int32)
    return pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("DIGIT", digit)])
