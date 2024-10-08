use crate::schema;
use crate::schema::payload_value;
use std::borrow;

#[derive(Clone, Debug)]
pub struct Args {
    args: Vec<schema::PayloadValue>,
}

#[derive(Clone, Debug)]
pub struct Kwargs<'k> {
    kwargs: Vec<(borrow::Cow<'k, str>, schema::PayloadValue)>,
}

impl Args {
    pub fn from_slice(slice: &[schema::PayloadValue]) -> Self {
        let args = slice.to_vec();
        Self { args }
    }

    pub fn into_list_value(self) -> schema::PayloadValue {
        let items = self.args;
        schema::PayloadValue {
            r#type: schema::ParameterType::ParamTypeList as i32,
            default_oneof: Some(payload_value::DefaultOneof::ListValue(
                schema::PayloadListValue { items },
            )),
        }
    }
}

impl<'k> Kwargs<'k> {
    pub fn from_slice(slice: &[(borrow::Cow<'k, str>, schema::PayloadValue)]) -> Self {
        let kwargs = slice.to_vec();
        Self { kwargs }
    }
}

#[macro_export]
macro_rules! args {
    [$($items:expr),*] => {
        $crate::arguments::Args::from_slice(&[
            $($items.into()),*
        ])
    };
}

#[macro_export]
macro_rules! kwargs {
    [$($key:ident = $value:expr),*] => {
        $crate::arguments::Kwargs::from_slice(&[
            $((stringify!($key).into(), $value.into())),*
        ])
    };
}
