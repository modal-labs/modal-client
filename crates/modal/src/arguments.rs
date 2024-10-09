use crate::schema;
use crate::schema::payload_value;
use std::borrow;

#[derive(Clone, Debug)]
pub struct CombinedArgs<'k> {
    pub args: Args,
    pub kwargs: Kwargs<'k>,
}

#[derive(Clone, Debug)]
pub struct Args(Vec<schema::PayloadValue>);

#[derive(Clone, Debug)]
pub struct Kwargs<'k>(Vec<(borrow::Cow<'k, str>, schema::PayloadValue)>);

impl<'k> CombinedArgs<'k> {
    pub fn new(args: Args, kwargs: Kwargs<'k>) -> Self {
        Self { args, kwargs }
    }
}

impl Args {
    pub fn from_slice(slice: &[schema::PayloadValue]) -> Self {
        Self(slice.to_vec())
    }

    pub fn into_list_value(self) -> schema::PayloadValue {
        let items = self.0;
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
        Self(slice.to_vec())
    }

    pub fn into_dict_value(self) -> schema::PayloadValue {
        let keys = self.0.iter().map(|(k, _)| k.clone().into_owned()).collect();
        let values = self.0.into_iter().map(|(_, v)| v.into()).collect();
        schema::PayloadValue {
            r#type: schema::ParameterType::ParamTypeDict as i32,
            default_oneof: Some(payload_value::DefaultOneof::DictValue(
                schema::PayloadDictValue { keys, values },
            )),
        }
    }
}

#[macro_export]
macro_rules! args {
    [$($rest:tt)*] => {
        $crate::args_builder!(rest: [$($rest)*], saw_kwarg: false, args: [], kwargs: []);
    };
}

#[macro_export]
macro_rules! args_builder {
    // TODO: can probably combine the "New"/"Last" cases

    // New kwarg case (rest starts with ident=expr)
    (rest: [$new_key:ident = $new_value:expr, $($rest:tt)*],
     saw_kwarg: $_:expr, // Allow kwargs any time
     args: [$($arg:expr),*],
     kwargs: [$(($key:expr, $value:expr)),*]
    ) => {
        $crate::args_builder!(
            rest: [$($rest)*],
            saw_kwarg: true,
            args: [$($arg),*],
            kwargs: [
                $(($key, $value),)*
                (::core::convert::From::from(stringify!($new_key)), ::core::convert::From::from($new_value))
            ]
        );
    };
    // Last kwarg case (rest only has one more kwarg)
    (rest: [$new_key:ident = $new_value:expr],
     saw_kwarg: $_:expr, // Allow kwargs any time
     args: [$($arg:expr),*],
     kwargs: [$(($key:expr, $value:expr)),*]) => {
        $crate::args_builder!(
            rest: [],
            saw_kwarg: true,
            args: [$($arg),*],
            kwargs: [
                $(($key, $value),)*
                (::core::convert::From::from(stringify!($new_key)), ::core::convert::From::from($new_value))
            ]
        );
    };
    // New arg case (rest starts with an expr)
    (rest: [$new_arg:expr, $($rest:tt)*],
     saw_kwarg: false, // Allow args only if we haven't seen kwargs yet
     args: [$($arg:expr),*],
     kwargs: [$(($key:expr, $value:expr)),*]
    ) => {
        $crate::args_builder!(
            rest: [$($rest)*],
            saw_kwarg: false,
            args: [$($arg,)* ::core::convert::From::from($new_arg)],
            kwargs: [$(($key, $value)),*]
        );
    };
    // Last arg case (rest only has one more kwarg)
    (rest: [$new_arg:expr],
     saw_kwarg: false, // Allow args only if we haven't seen kwargs yet
     args: [$($arg:expr),*],
     kwargs: [$(($key:expr, $value:expr)),*]
    ) => {
        $crate::args_builder!(
            rest: [],
            saw_kwarg: false,
            args: [$($arg,)* ::core::convert::From::from($new_arg)],
            kwargs: [$(($key, $value)),*]
        );
    };
    // Base case: done processing rest
    (rest: [],
     saw_kwarg: $_:expr, // Don't care
     args: [$($arg:expr),*],
     kwargs: [$(($key:expr, $value:expr)),*]
    ) => {
        $crate::arguments::CombinedArgs::new(
            $crate::arguments::Args::from_slice(&[$($arg),*]),
            $crate::arguments::Kwargs::from_slice(&[$(($key, $value)),*])
        );
    };
}
