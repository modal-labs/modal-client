use crate::schema::function_input;
use crate::{arguments, auth, schema};
use prost::Message;
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
    pub async fn call<'k>(
        &mut self,
        args: arguments::Args,
        kwargs: arguments::Kwargs<'k>,
    ) -> anyhow::Result<()> {
        let combined: schema::PayloadValue =
            vec![args.into_list_value(), kwargs.into_dict_value()].into();
        let input = schema::FunctionPutInputsItem {
            idx: 0,
            input: Some(schema::FunctionInput {
                final_input: true,
                data_format: schema::DataFormat::PayloadValue as i32,
                method_name: None,
                args_oneof: Some(function_input::ArgsOneof::Args(
                    combined.encode_to_vec().into(),
                )),
            }),
        };

        let func_map = self
            .client
            .function_map(schema::FunctionMapRequest {
                function_id: self.id.clone(),
                function_call_type: schema::FunctionCallType::Unary as i32,
                pipelined_inputs: vec![input],
                function_call_invocation_type: schema::FunctionCallInvocationType::SyncLegacy
                    as i32,
                ..Default::default()
            })
            .await?
            .into_inner();
        dbg!(&func_map);

        let fibs = Fibonacci::new();
        for (attempt, fib) in fibs.enumerate() {
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
            dbg!(&attempt, &func_outputs);

            if func_outputs.num_unfinished_inputs == 0 {
                break;
            }

            tokio::time::sleep(time::Duration::from_millis(10 * fib)).await;
        }
        Ok(())
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
