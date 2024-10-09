use crate::schema::function_input;
use crate::{arguments, auth, schema, value};
use prost::Message;
use schema::generic_result;
use std::time;

#[derive(Clone, Debug)]
pub struct Function {
    id: String,
    client: auth::AuthedClient,
}

struct Fibonacci {
    a: u64,
    b: u64,
}

impl Function {
    pub fn new(id: String, client: auth::AuthedClient) -> Self {
        Self { id, client }
    }
}

impl Function {
    pub async fn call(&mut self, args: arguments::CombinedArgs) -> anyhow::Result<value::Value> {
        let args: value::Value = args.into();
        let input = schema::FunctionPutInputsItem {
            idx: 0,
            input: Some(schema::FunctionInput {
                final_input: true,
                data_format: schema::DataFormat::PayloadValue as i32,
                method_name: None,
                args_oneof: Some(function_input::ArgsOneof::Args(
                    args.into_proto().encode_to_vec().into(),
                )),
            }),
        };

        let func_map = self
            .client
            .function_map(schema::FunctionMapRequest {
                function_id: self.id.clone(),
                function_call_type: schema::FunctionCallType::Unary as i32,
                pipelined_inputs: vec![input],
                function_call_invocation_type: schema::FunctionCallInvocationType::Async as i32,
                ..Default::default()
            })
            .await?
            .into_inner();

        let mut fibs = Fibonacci::new();
        let func_outputs = loop {
            let fib = fibs.next().expect("fibonacci sequence is infinite");
            let func_outputs = self
                .client
                .function_get_outputs(schema::FunctionGetOutputsRequest {
                    function_call_id: func_map.function_call_id.clone(),
                    last_entry_id: "0-0".to_owned(),
                    requested_at: time::SystemTime::now()
                        .duration_since(time::UNIX_EPOCH)?
                        .as_secs_f64(),
                    ..Default::default()
                })
                .await?
                .into_inner();

            if func_outputs.num_unfinished_inputs == 0 {
                break func_outputs;
            }

            tokio::time::sleep(time::Duration::from_millis(10 * fib)).await;
        };

        let output = func_outputs
            .outputs
            .into_iter()
            .next()
            .expect("exactly one output from function");
        let result = output.result.unwrap_or_default();
        if result.status == generic_result::GenericStatus::Success as i32 {
            // TODO: handle other cases
            assert_eq!(schema::DataFormat::PayloadValue as i32, output.data_format);
            match result.data_oneof {
                Some(generic_result::DataOneof::Data(bytes)) => {
                    let proto = schema::PayloadValue::decode(&*bytes)?;
                    Ok(value::Value::from_proto(proto)
                        .ok_or_else(|| anyhow::anyhow!("could not decode response proto"))?)
                }
                other => Err(anyhow::anyhow!(
                    "could not decode response proto: {:?}",
                    other
                )),
            }
        } else {
            let status = generic_result::GenericStatus::try_from(result.status)
                .unwrap_or(generic_result::GenericStatus::Unspecified);
            let exception = result.exception;
            Err(anyhow::anyhow!(
                "function call failed, status: {status:?}\nexception: {exception}"
            ))
        }
    }
}

impl Fibonacci {
    fn new() -> Self {
        Fibonacci { a: 1, b: 0 }
    }
}
impl Iterator for Fibonacci {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let r = self.b;
        self.b = self.a;
        self.a += r;
        Some(r)
    }
}
