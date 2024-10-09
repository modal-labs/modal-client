use crate::schema::payload_value;
use crate::{arguments, schema};

macro_rules! impl_payload_value_int {
    ($int_ty:ty) => {
        impl From<$int_ty> for schema::PayloadValue {
            fn from(value: $int_ty) -> Self {
                schema::PayloadValue {
                    r#type: schema::ParameterType::ParamTypeInt as i32,
                    // TODO: handle cast from u64 to i64 in a more fail-safe manner
                    default_oneof: Some(payload_value::DefaultOneof::IntValue(value as i64)),
                }
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

impl<A> From<&A> for schema::PayloadValue
where
    A: Into<schema::PayloadValue>,
{
    fn from(value: &A) -> Self {
        value.into()
    }
}
impl From<&str> for schema::PayloadValue {
    fn from(value: &str) -> Self {
        schema::PayloadValue {
            r#type: schema::ParameterType::ParamTypeString as i32,
            default_oneof: Some(payload_value::DefaultOneof::StrValue(value.to_owned())),
        }
    }
}

impl From<String> for schema::PayloadValue {
    fn from(value: String) -> Self {
        schema::PayloadValue {
            r#type: schema::ParameterType::ParamTypeString as i32,
            default_oneof: Some(payload_value::DefaultOneof::StrValue(value)),
        }
    }
}

impl From<bool> for schema::PayloadValue {
    fn from(value: bool) -> Self {
        schema::PayloadValue {
            r#type: schema::ParameterType::ParamTypeBool as i32,
            default_oneof: Some(payload_value::DefaultOneof::BoolValue(value)),
        }
    }
}
impl From<f32> for schema::PayloadValue {
    fn from(value: f32) -> Self {
        schema::PayloadValue {
            r#type: schema::ParameterType::ParamTypeFloat as i32,
            default_oneof: Some(payload_value::DefaultOneof::FloatValue(value)),
        }
    }
}

impl From<bytes::Bytes> for schema::PayloadValue {
    fn from(value: bytes::Bytes) -> Self {
        schema::PayloadValue {
            r#type: schema::ParameterType::ParamTypeBytes as i32,
            default_oneof: Some(payload_value::DefaultOneof::BytesValue(value)),
        }
    }
}

impl<A> From<Vec<A>> for schema::PayloadValue
where
    A: Into<schema::PayloadValue>,
{
    fn from(value: Vec<A>) -> Self {
        let items = value.into_iter().map(Into::into).collect();
        schema::PayloadValue {
            r#type: schema::ParameterType::ParamTypeList as i32,
            default_oneof: Some(payload_value::DefaultOneof::ListValue(
                schema::PayloadListValue { items },
            )),
        }
    }
}

impl<A> From<&[A]> for schema::PayloadValue
where
    A: Into<schema::PayloadValue>,
{
    fn from(value: &[A]) -> Self {
        let items = value.iter().map(Into::into).collect();
        schema::PayloadValue {
            r#type: schema::ParameterType::ParamTypeList as i32,
            default_oneof: Some(payload_value::DefaultOneof::ListValue(
                schema::PayloadListValue { items },
            )),
        }
    }
}

impl<K, A> From<Vec<(K, A)>> for schema::PayloadValue
where
    K: ToOwned<Owned = String>,
    A: Into<schema::PayloadValue>,
{
    fn from(value: Vec<(K, A)>) -> Self {
        let keys = value.iter().map(|(k, _)| k.to_owned()).collect();
        let values = value.into_iter().map(|(_, v)| v.into()).collect();
        schema::PayloadValue {
            r#type: schema::ParameterType::ParamTypeDict as i32,
            default_oneof: Some(payload_value::DefaultOneof::DictValue(
                schema::PayloadDictValue { keys, values },
            )),
        }
    }
}

impl<K, A> From<&[(K, A)]> for schema::PayloadValue
where
    K: ToOwned<Owned = String>,
    A: Into<schema::PayloadValue>,
{
    fn from(value: &[(K, A)]) -> Self {
        let keys = value.iter().map(|(k, _)| k.to_owned()).collect();
        let values = value.into_iter().map(|(_, v)| v.into()).collect();
        schema::PayloadValue {
            r#type: schema::ParameterType::ParamTypeDict as i32,
            default_oneof: Some(payload_value::DefaultOneof::DictValue(
                schema::PayloadDictValue { keys, values },
            )),
        }
    }
}

impl From<arguments::Args> for schema::PayloadValue {
    fn from(value: arguments::Args) -> Self {
        value.into_list_value()
    }
}

impl From<arguments::Kwargs> for schema::PayloadValue {
    fn from(value: arguments::Kwargs) -> Self {
        value.into_dict_value()
    }
}
