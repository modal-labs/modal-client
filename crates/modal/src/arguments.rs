use crate::value;
use std::borrow;

#[derive(Clone, Debug)]
pub struct CombinedArgs {
    args: Args,
    kwargs: Kwargs,
}

#[derive(Clone, Debug)]
pub struct Args(Vec<value::Value>);

#[derive(Clone, Debug)]
pub struct Kwargs(Vec<(borrow::Cow<'static, str>, value::Value)>);

impl CombinedArgs {
    pub fn new(args: Args, kwargs: Kwargs) -> Self {
        Self { args, kwargs }
    }

    pub fn into_parts(self) -> (Args, Kwargs) {
        (self.args, self.kwargs)
    }

    pub fn args(&self) -> &Args {
        &self.args
    }

    pub fn arg(&self, index: usize) -> Option<&value::Value> {
        self.args.values().get(index)
    }

    pub fn kwargs(&self) -> &Kwargs {
        &self.kwargs
    }

    pub fn kwarg(&self, key: &str) -> Option<&value::Value> {
        self.kwargs
            .key_value_pairs()
            .iter()
            .find_map(|(k, v)| if k == key { Some(v) } else { None })
    }
}

impl Args {
    pub fn from_slice(slice: &[value::Value]) -> Self {
        Self(slice.to_vec())
    }

    pub fn values(&self) -> &[value::Value] {
        &self.0
    }

    pub fn into_list_value(self) -> value::Value {
        value::Value::List(self.0)
    }
}

impl Kwargs {
    pub fn from_slice(slice: &[(borrow::Cow<'static, str>, value::Value)]) -> Self {
        Self(slice.to_vec())
    }

    pub fn key_value_pairs(&self) -> &[(borrow::Cow<'static, str>, value::Value)] {
        &self.0
    }

    pub fn into_dict_value(self) -> value::Value {
        value::Value::Dict(self.0)
    }
}

/// Construct args to be passed to `Function::call`
///
/// # Examples
///
/// Passing normal args:
///
/// ```
/// use modal::{args, Value};
/// let args = args!(1, 2, "3", true);
/// assert_eq!(&Value::Integer(1), args.arg(0).unwrap());
/// assert_eq!(&Value::Integer(2), args.arg(1).unwrap());
/// assert_eq!(&Value::String("3".to_owned()), args.arg(2).unwrap());
/// assert_eq!(&Value::Boolean(true), args.arg(3).unwrap());
/// ```
///
/// Passing kwargs:
///
/// ```
/// use modal::{args, Value};
/// let args = args!(1, 2, foo="3", bar=true);
/// assert_eq!(&Value::Integer(1), args.arg(0).unwrap());
/// assert_eq!(&Value::Integer(2), args.arg(1).unwrap());
/// assert_eq!(&Value::String("3".to_owned()), args.kwarg("foo").unwrap());
/// assert_eq!(&Value::Boolean(true), args.kwarg("bar").unwrap());
/// ```
///
/// It is not allowed to pass args after kwargs:
///
/// ```compile_fail
/// use modal::args;
/// // Compile error: Cannot pass a normal (positional) arg after named kwargs
/// let _ = args!(foo="3", bar=true, 1, 2);
/// ```
#[macro_export]
macro_rules! args {
    [$($rest:tt)*] => {
        $crate::args_builder!(rest: [$($rest)*], saw_kwarg: false, args: [], kwargs: [])
    };
}

#[macro_export]
macro_rules! args_builder {
    // New kwarg case (rest starts with ident=expr)
    (rest: [$new_key:ident = $new_value:expr $(,$($rest:tt)*)?],
     saw_kwarg: $_:expr, // Allow kwargs any time
     args: [$($arg:expr),*],
     kwargs: [$(($key:expr, $value:expr)),*]
    ) => {
        $crate::args_builder!(
            rest: [$($($rest)*)?],
            saw_kwarg: true,
            args: [$($arg),*],
            kwargs: [
                $(($key, $value),)*
                (::core::convert::From::from(stringify!($new_key)), ::core::convert::From::from($new_value))
            ]
        )
    };
    // New arg case (rest starts with an expr)
    (rest: [$new_arg:expr $(,$($rest:tt)*)?],
     saw_kwarg: false, // Allow args only if we haven't seen kwargs yet
     args: [$($arg:expr),*],
     kwargs: [$(($key:expr, $value:expr)),*]
    ) => {
        $crate::args_builder!(
            rest: [$($($rest)*)?],
            saw_kwarg: false,
            args: [$($arg,)* ::core::convert::From::from($new_arg)],
            kwargs: [$(($key, $value)),*]
        )
    };
    // Report a nicer compile error if an arg follows a kwarg
    (rest: [$new_arg:expr $(,$($rest:tt)*)?],
     saw_kwarg: true,
     args: [$($arg:expr),*],
     kwargs: [$(($key:expr, $value:expr)),*]
    ) => {
        compile_error!("Cannot pass a normal (positional) arg after named kwargs");
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
        )
    };
}

#[test]
fn args_simple() {
    let args = args!(1, "2", true, [4, 5].as_slice());
    // Easier to assert against debug string than to construct big expected CombinedArgs
    let actual = format!("{:?}", args);
    let expected = "CombinedArgs { \
        args: Args([Integer(1), String(\"2\"), Boolean(true), List([Integer(4), Integer(5)])]), \
        kwargs: Kwargs([]) \
    }";
    assert_eq!(expected, &actual);
}

#[test]
fn args_kwargs() {
    let args = args!(1, "2", foo = true, bar = [4, 5].as_slice());
    // Easier to assert against debug string than to construct big expected CombinedArgs
    let actual = format!("{:?}", args);
    let expected = "CombinedArgs { \
        args: Args([Integer(1), String(\"2\")]), \
        kwargs: Kwargs([(\"foo\", Boolean(true)), (\"bar\", List([Integer(4), Integer(5)]))]) \
    }";
    assert_eq!(expected, &actual);
}
