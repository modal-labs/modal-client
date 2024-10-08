use crate::schema::function_input;
use crate::{arguments, auth, schema};
use prost::Message;

#[derive(Clone, Debug)]
pub struct Function {
    id: String,
    client: auth::AuthedClient,
}

impl Function {
    pub fn new(id: String, client: auth::AuthedClient) -> Self {
        Self { id, client }
    }
}

impl Function {
    pub async fn call<'k>(
        &mut self,
        args: arguments::Args,
        kwargs: arguments::Kwargs<'k>,
    ) -> anyhow::Result<()> {
        // TODO: discuss how we want to encode args + kwargs... list of [list, dict]?
        let input = schema::FunctionPutInputsItem {
            idx: 0,
            input: Some(schema::FunctionInput {
                final_input: true,
                data_format: schema::DataFormat::PayloadValue as i32,
                method_name: None,
                args_oneof: Some(function_input::ArgsOneof::Args(
                    args.into_list_value().encode_to_vec().into(),
                )),
            }),
        };
        self.client
            .function_map(schema::FunctionMapRequest {
                function_id: self.id.clone(),
                parent_input_id: "".to_owned(),
                return_exceptions: false,
                function_call_type: schema::FunctionCallType::Unary as i32,
                pipelined_inputs: vec![input],
                function_call_invocation_type: schema::FunctionCallInvocationType::Unspecified
                    as i32,
            })
            .await?;
        Ok(())
    }
}
