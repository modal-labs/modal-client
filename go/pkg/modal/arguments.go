package modal

import (
	"fmt"
	"reflect"
)

type CombinedArgs struct {
	args   Args
	kwargs Kwargs
}

type Args []Value
type Kwargs map[string]Value

func NewCombinedArgs(args Args, kwargs Kwargs) CombinedArgs {
	return CombinedArgs{args: args, kwargs: kwargs}
}

func (ca CombinedArgs) IntoParts() (Args, Kwargs) {
	return ca.args, ca.kwargs
}

func (ca CombinedArgs) Args() Args {
	return ca.args
}

func (ca CombinedArgs) Arg(index int) *Value {
	if index < 0 || index >= len(ca.args) {
		return nil
	}
	return &ca.args[index]
}

func (ca CombinedArgs) Kwargs() Kwargs {
	return ca.kwargs
}

func (ca CombinedArgs) Kwarg(key string) *Value {
	if v, ok := ca.kwargs[key]; ok {
		return &v
	}
	return nil
}

func (ca CombinedArgs) ToProto() (*PayloadValue, error) {
	argsList := make([]*PayloadValue, len(ca.args))
	for i, arg := range ca.args {
		argProto := arg.ToProto()
		argsList[i] = argProto
	}

	kwargsDict := make(map[string]*PayloadValue)
	for k, v := range ca.kwargs {
		kwargProto := v.ToProto()
		kwargsDict[k] = kwargProto
	}

	return &PayloadValue{
		Type: ParameterType_PARAM_TYPE_LIST,
		DefaultOneof: &PayloadValue_ListValue{
			ListValue: &PayloadListValue{
				Items: []*PayloadValue{
					{
						Type: ParameterType_PARAM_TYPE_LIST,
						DefaultOneof: &PayloadValue_ListValue{
							ListValue: &PayloadListValue{
								Items: argsList,
							},
						},
					},
					{
						Type: ParameterType_PARAM_TYPE_DICT,
						DefaultOneof: &PayloadValue_DictValue{
							DictValue: &PayloadDictValue{
								Keys:   make([]string, 0, len(kwargsDict)),
								Values: make([]*PayloadValue, 0, len(kwargsDict)),
							},
						},
					},
				},
			},
		},
	}, nil
}

func (a Args) Values() []Value {
	return a
}

func (a Args) IntoListValue() Value {
	return Value{List: a}
}

func (k Kwargs) KeyValuePairs() map[string]Value {
	return k
}

func (k Kwargs) IntoDictValue() Value {
	return Value{Dict: k}
}

// NewArgs creates a new CombinedArgs from a mix of positional and keyword arguments
// Usage example:
//
//	args := NewArgs(1, "2", true, []int{4, 5}, struct{K,V interface{}}{K: "foo", V: true},)
func NewArgs(args ...interface{}) CombinedArgs {
	var positional Args
	kwargs := make(Kwargs)
	inKwargs := false

	for i := 0; i < len(args); i++ {
		arg := args[i]
		if kv, ok := arg.(struct{ K, V interface{} }); ok {
			inKwargs = true
			key, ok := kv.K.(string)
			if !ok {
				panic(fmt.Sprintf("Keyword argument key must be a string, got %T", kv.K))
			}
			kwargs[key] = FromInterface(kv.V)
		} else {
			if inKwargs {
				panic("Cannot pass a normal (positional) arg after named kwargs")
			}
			positional = append(positional, FromInterface(arg))
		}
	}

	return NewCombinedArgs(positional, kwargs)
}

// FromInterface uses Go generics to create a typed generic interface.
func FromInterface[T any](v T) Value {
	switch val := any(v).(type) {
	case string:
		return Value{String: &val}
	case int:
		i64 := int64(val)
		return Value{Integer: &i64}
	case int64:
		return Value{Integer: &val}
	case bool:
		return Value{Boolean: &val}
	case float32:
		return Value{Float: &val}
	case []byte:
		return Value{Bytes: val}
	case []any:
		list := make([]Value, len(val))
		for i, item := range val {
			list[i] = FromInterface(item)
		}
		return Value{List: list}
	case map[string]any:
		dict := make(map[string]Value)
		for k, v := range val {
			dict[k] = FromInterface(v)
		}
		return Value{Dict: dict}
	default:
		// Handle other types or panic
		panic(fmt.Sprintf("Unsupported type: %v", reflect.TypeOf(v)))
	}
}
