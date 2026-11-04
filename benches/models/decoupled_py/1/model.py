import json

import numpy as np
import triton_python_backend_utils as pb_utils

FINAL = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        self.target = config["parameters"]["target"]["string_value"]

    def execute(self, requests):
        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "Input3").as_numpy()
            sender = request.get_response_sender()
            for image in images:
                result = pb_utils.InferenceRequest(
                    model_name=self.target,
                    requested_output_names=["Plus214_Output_0"],
                    inputs=[pb_utils.Tensor("Input3", image[np.newaxis])],
                ).exec()
                if result.has_error():
                    sender.send(pb_utils.InferenceResponse(error=result.error()), flags=FINAL)
                    break
                logits = pb_utils.get_output_tensor_by_name(result, "Plus214_Output_0").as_numpy()
                digit = np.array([[int(np.argmax(logits))]], dtype=np.int32)
                sender.send(
                    pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("DIGIT", digit)])
                )
            else:
                sender.send(flags=FINAL)
        return None
