package modal

import (
	"bytes"
	"encoding/json"
)

type Value struct {
	String  *string
	Integer *int64
	Boolean *bool
	Float   *float32
	Bytes   []byte
	List    []Value
	Dict    map[string]Value
}

func (v Value) Type() ParameterType {
	switch {
	case v.String != nil:
		return ParameterType_PARAM_TYPE_STRING
	case v.Integer != nil:
		return ParameterType_PARAM_TYPE_INT
	case v.Boolean != nil:
		return ParameterType_PARAM_TYPE_BOOL
	case v.Float != nil:
		return ParameterType_PARAM_TYPE_FLOAT
	case v.Bytes != nil:
		return ParameterType_PARAM_TYPE_BYTES
	case v.List != nil:
		return ParameterType_PARAM_TYPE_LIST
	case v.Dict != nil:
		return ParameterType_PARAM_TYPE_DICT
	default:
		return ParameterType_PARAM_TYPE_UNSPECIFIED
	}
}

func (v Value) MarshalJSON() ([]byte, error) {
	switch v.Type() {
	case ParameterType_PARAM_TYPE_STRING:
		return json.Marshal(v.String)
	case ParameterType_PARAM_TYPE_INT:
		return json.Marshal(v.Integer)
	case ParameterType_PARAM_TYPE_BOOL:
		return json.Marshal(v.Boolean)
	case ParameterType_PARAM_TYPE_FLOAT:
		return json.Marshal(v.Float)
	case ParameterType_PARAM_TYPE_BYTES:
		return json.Marshal(v.Bytes)
	case ParameterType_PARAM_TYPE_LIST:
		return json.Marshal(v.List)
	case ParameterType_PARAM_TYPE_DICT:
		return json.Marshal(v.Dict)
	default:
		return []byte("null"), nil
	}
}

func (v *Value) UnmarshalJSON(data []byte) error {
	if bytes.Equal(data, []byte("null")) {
		return nil
	}

	var temp interface{}
	if err := json.Unmarshal(data, &temp); err != nil {
		return err
	}

	switch val := temp.(type) {
	case string:
		v.String = &val
	case float64:
		if val == float64(int64(val)) {
			i := int64(val)
			v.Integer = &i
		} else {
			f := float32(val)
			v.Float = &f
		}
	case bool:
		v.Boolean = &val
	case []interface{}:
		v.List = make([]Value, len(val))
		for i, item := range val {
			itemBytes, _ := json.Marshal(item)
			json.Unmarshal(itemBytes, &v.List[i])
		}
	case map[string]interface{}:
		v.Dict = make(map[string]Value)
		for key, item := range val {
			itemBytes, _ := json.Marshal(item)
			var itemValue Value
			json.Unmarshal(itemBytes, &itemValue)
			v.Dict[key] = itemValue
		}
	}
	return nil
}

func FromProto(value *PayloadValue) *Value {
	switch value.DefaultOneof.(type) {
	case *PayloadValue_StrValue:
		strValue := value.GetStrValue()
		return &Value{String: &strValue}
	case *PayloadValue_IntValue:
		intValue := value.GetIntValue()
		return &Value{Integer: &intValue}
	case *PayloadValue_BoolValue:
		boolValue := value.GetBoolValue()
		return &Value{Boolean: &boolValue}
	case *PayloadValue_FloatValue:
		floatValue := value.GetFloatValue()
		return &Value{Float: &floatValue}
	case *PayloadValue_BytesValue:
		return &Value{Bytes: value.GetBytesValue()}
	case *PayloadValue_ListValue:
		list := value.GetListValue()
		v := &Value{List: make([]Value, len(list.Items))}
		for i, item := range list.Items {
			if itemValue := FromProto(item); itemValue != nil {
				v.List[i] = *itemValue
			}
		}
		return v
	case *PayloadValue_DictValue:
		dict := value.GetDictValue()
		v := &Value{Dict: make(map[string]Value)}
		for i, key := range dict.Keys {
			if itemValue := FromProto(dict.Values[i]); itemValue != nil {
				v.Dict[key] = *itemValue
			}
		}
		return v
	default:
		return nil
	}
}

func (v Value) ToProto() *PayloadValue {
	pv := &PayloadValue{Type: v.Type()}
	switch v.Type() {
	case ParameterType_PARAM_TYPE_STRING:
		pv.DefaultOneof = &PayloadValue_StrValue{StrValue: *v.String}
	case ParameterType_PARAM_TYPE_INT:
		pv.DefaultOneof = &PayloadValue_IntValue{IntValue: *v.Integer}
	case ParameterType_PARAM_TYPE_BOOL:
		pv.DefaultOneof = &PayloadValue_BoolValue{BoolValue: *v.Boolean}
	case ParameterType_PARAM_TYPE_FLOAT:
		pv.DefaultOneof = &PayloadValue_FloatValue{FloatValue: *v.Float}
	case ParameterType_PARAM_TYPE_BYTES:
		pv.DefaultOneof = &PayloadValue_BytesValue{BytesValue: v.Bytes}
	case ParameterType_PARAM_TYPE_LIST:
		items := make([]*PayloadValue, len(v.List))
		for i, item := range v.List {
			items[i] = item.ToProto()
		}
		pv.DefaultOneof = &PayloadValue_ListValue{ListValue: &PayloadListValue{Items: items}}
	case ParameterType_PARAM_TYPE_DICT:
		keys := make([]string, 0, len(v.Dict))
		values := make([]*PayloadValue, 0, len(v.Dict))
		for key, value := range v.Dict {
			keys = append(keys, key)
			values = append(values, value.ToProto())
		}
		pv.DefaultOneof = &PayloadValue_DictValue{DictValue: &PayloadDictValue{Keys: keys, Values: values}}
	}
	return pv
}
