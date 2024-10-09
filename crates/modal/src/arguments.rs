use crate::value;
use std::borrow;

#[derive(Clone, Debug)]
pub struct CombinedArgs {
    pub args: Args,
    pub kwargs: Kwargs,
}

#[derive(Clone, Debug)]
pub struct Args(Vec<value::Value>);

#[derive(Clone, Debug)]
pub struct Kwargs(Vec<(borrow::Cow<'static, str>, value::Value)>);

impl CombinedArgs {
    pub fn new(args: Args, kwargs: Kwargs) -> Self {
        Self { args, kwargs }
    }
}

impl Args {
    pub fn from_slice(slice: &[value::Value]) -> Self {
        Self(slice.to_vec())
    }

    pub fn into_list_value(self) -> value::Value {
        value::Value::List(self.0)
    }
}

impl Kwargs {
    pub fn from_slice(slice: &[(borrow::Cow<'static, str>, value::Value)]) -> Self {
        Self(slice.to_vec())
    }

    pub fn into_dict_value(self) -> value::Value {
        value::Value::Dict(self.0)
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
