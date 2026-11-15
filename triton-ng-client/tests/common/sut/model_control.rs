use crate::common::support::string;
use crate::common::sut::Sut;
use anyhow::Result;
use tonic::Code;

impl Sut {
    pub(crate) async fn model_control(
        &self,
        parameters: &[(&str, &str)],
    ) -> Result<std::result::Result<(), Code>> {
        let config = &self.config;

        let result = self
            .infer_raw_with_parameters(
                &config.backend_model,
                vec![config.mnist_blank_input()],
                &[&config.backend_output_name],
                parameters
                    .iter()
                    .map(|(name, value)| (name.to_string(), string(value)))
                    .collect(),
            )
            .await;
        match result {
            Ok(_) => Ok(Ok(())),
            Err(error) => match error.downcast_ref::<triton_ng_client::Error>() {
                Some(triton_ng_client::Error::Grpc(status)) => Ok(Err(status.code())),
                _ => return Err(anyhow::anyhow!("not a gRPC error: {error:#}")),
            },
        }
    }
}
