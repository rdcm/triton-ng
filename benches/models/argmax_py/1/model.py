import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            image = pb_utils.get_input_tensor_by_name(request, "Input3").as_numpy()
            digit = np.array([[int(np.argmax(image))]], dtype=np.int32)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("DIGIT", digit)])
            )
        return responses
