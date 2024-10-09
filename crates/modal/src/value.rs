use crate::schema::payload_value;
use crate::{arguments, schema};
use std::borrow;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Value {
    String(String),
    Integer(i64),
    Boolean(bool),
    Float(f32),
    Bytes(bytes::Bytes),
    List(Vec<Value>),
    Dict(Vec<(borrow::Cow<'static, str>, Value)>),
}

impl Value {
    pub(crate) fn from_proto(value: schema::PayloadValue) -> Option<Self> {
        match value.default_oneof {
            None => None,
            Some(payload_value::DefaultOneof::StrValue(v)) => Some(Value::String(v)),
            Some(payload_value::DefaultOneof::IntValue(v)) => Some(Value::Integer(v)),
            Some(payload_value::DefaultOneof::BoolValue(v)) => Some(Value::Boolean(v)),
            Some(payload_value::DefaultOneof::FloatValue(v)) => Some(Value::Float(v)),
            Some(payload_value::DefaultOneof::BytesValue(v)) => Some(Value::Bytes(v)),
            Some(payload_value::DefaultOneof::ListValue(list)) => {
                let mut vec = Vec::with_capacity(list.items.len());
                for item in list.items {
                    vec.push(Value::from_proto(item)?);
                }
                Some(Value::List(vec))
            }
            Some(payload_value::DefaultOneof::DictValue(dict)) => {
                let len = dict.keys.len();
                let pairs = dict.keys.into_iter().zip(dict.values.into_iter());
                let mut vec = Vec::with_capacity(len);

                for (k, v) in pairs {
                    vec.push((k.into(), Value::from_proto(v)?));
                }

                Some(Value::Dict(vec))
            }
            Some(payload_value::DefaultOneof::PickleValue(_)) => None,
        }
    }
}

impl Value {
    pub fn into_proto(self) -> schema::PayloadValue {
        match self {
            Value::String(v) => schema::PayloadValue {
                r#type: schema::ParameterType::ParamTypeString as i32,
                default_oneof: Some(payload_value::DefaultOneof::StrValue(v)),
            },
            Value::Integer(v) => schema::PayloadValue {
                r#type: schema::ParameterType::ParamTypeInt as i32,
                default_oneof: Some(payload_value::DefaultOneof::IntValue(v)),
            },
            Value::Boolean(v) => schema::PayloadValue {
                r#type: schema::ParameterType::ParamTypeBool as i32,
                default_oneof: Some(payload_value::DefaultOneof::BoolValue(v)),
            },
            Value::Float(v) => schema::PayloadValue {
                r#type: schema::ParameterType::ParamTypeFloat as i32,
                default_oneof: Some(payload_value::DefaultOneof::FloatValue(v)),
            },
            Value::Bytes(v) => schema::PayloadValue {
                r#type: schema::ParameterType::ParamTypeBytes as i32,
                default_oneof: Some(payload_value::DefaultOneof::BytesValue(v)),
            },
            Value::List(v) => {
                let items = v.into_iter().map(|v| v.into_proto()).collect();
                schema::PayloadValue {
                    r#type: schema::ParameterType::ParamTypeList as i32,
                    default_oneof: Some(payload_value::DefaultOneof::ListValue(
                        schema::PayloadListValue { items },
                    )),
                }
            }
            Value::Dict(v) => {
                let keys = v.iter().map(|(k, _)| k.clone().into_owned()).collect();
                let values = v.into_iter().map(|(_, v)| v.into_proto()).collect();
                schema::PayloadValue {
                    r#type: schema::ParameterType::ParamTypeDict as i32,
                    default_oneof: Some(payload_value::DefaultOneof::DictValue(
                        schema::PayloadDictValue { keys, values },
                    )),
                }
            }
        }
    }
}

macro_rules! impl_payload_value_int {
    ($int_ty:ty) => {
        impl From<$int_ty> for Value {
            fn from(value: $int_ty) -> Self {
                // TODO: handle cast from u64 to i64 in a more fail-safe manner
                Value::Integer(value as i64)
            }
        }
    };
}

macro_rules! impl_payload_value_ints {
    () => {};
    ($int_ty:ty) => {
        impl_payload_value_int!($int_ty);
    };
    ($int_ty:ty, $($int_tys:ty),*) => {
        impl_payload_value_int!($int_ty);
        impl_payload_value_ints!($($int_tys),*);
    };
}

impl_payload_value_ints!(u8, u16, u32, u64, i8, i16, i32, i64);

impl<A> From<&A> for Value
where
    A: Into<Value> + Clone,
{
    fn from(value: &A) -> Self {
        value.clone().into()
    }
}
impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::String(value.to_owned())
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::String(value)
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::Boolean(value)
    }
}
impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Value::Float(value)
    }
}

impl From<bytes::Bytes> for Value {
    fn from(value: bytes::Bytes) -> Self {
        Value::Bytes(value)
    }
}

impl<A> From<Vec<A>> for Value
where
    A: Into<Value>,
{
    fn from(value: Vec<A>) -> Self {
        let items = value.into_iter().map(Into::into).collect();
        Value::List(items)
    }
}

impl<A> From<&[A]> for Value
where
    A: Into<Value> + Clone,
{
    fn from(value: &[A]) -> Self {
        let items = value.iter().map(Into::into).collect();
        Value::List(items)
    }
}

impl<K, A> From<Vec<(K, A)>> for Value
where
    K: ToOwned<Owned = String>,
    A: Into<Value>,
{
    fn from(value: Vec<(K, A)>) -> Value {
        Value::Dict(
            value
                .into_iter()
                .map(|(k, v)| (k.to_owned().into(), v.into()))
                .collect(),
        )
    }
}

impl<K, A> From<&[(K, A)]> for Value
where
    K: ToOwned<Owned = String>,
    A: Into<Value> + Clone,
{
    fn from(value: &[(K, A)]) -> Self {
        Value::Dict(
            value
                .iter()
                .map(|(k, v)| (k.to_owned().into(), v.into()))
                .collect(),
        )
    }
}

impl From<arguments::Args> for Value {
    fn from(value: arguments::Args) -> Self {
        value.into_list_value()
    }
}

impl From<arguments::Kwargs> for Value {
    fn from(value: arguments::Kwargs) -> Self {
        value.into_dict_value()
    }
}

impl From<arguments::CombinedArgs> for Value {
    fn from(value: arguments::CombinedArgs) -> Self {
        let (args, kwargs) = value.into_parts();
        let items = vec![args.into_list_value(), kwargs.into_dict_value()];
        Value::List(items)
    }
}
