class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = ConfigDict(v)
    
    """配置字典，支持点号访问"""
    def __getattr__(self, key):
        # try:
        #     return self[key]
        # except KeyError:
        #     # 让getattr的默认值机制生效
        #     raise AttributeError(key)
        return self.get(key, None)
        
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys()) + list(super().__dir__())
    
    def check_schema(self, schema: dict):
        """
        用 JSON Schema（dict 形式）校验当前 ConfigDict。
        失败时抛 ValueError / TypeError，信息中包含字段路径。
        """
        def _check(value, schema, path):
            def _err(msg):
                raise ValueError(f"{path}: {msg}")

            type_req = schema.get("type")
            if type_req == "object":
                if not isinstance(value, dict):
                    _err(f"expected object, got {type(value).__name__}")
            elif type_req == "array":
                if not isinstance(value, list):
                    _err(f"expected array, got {type(value).__name__}")
            elif type_req in ("string", "integer", "number", "boolean"):
                py_type = {"string": str, "integer": int,
                           "number": (int, float), "boolean": bool}[type_req]
                if not isinstance(value, py_type):
                    _err(f"expected {type_req}, got {type(value).__name__}")

            # ---- object 专用检查 ----
            if type_req == "object":
                props       = schema.get("properties", {})
                required    = schema.get("required", [])
                allow_extra = schema.get("additionalProperties", True)

                for k in required:
                    if k not in value:
                        _err(f"missing required field '{k}'")

                if allow_extra is False:
                    extra = set(value) - set(props)
                    if extra:
                        _err(f"unexpected fields {list(extra)}")

                for k, sub_schema in props.items():
                    if k in value:
                        child_path = f"{path}.{k}" if path != "<root>" else k
                        _check(value[k], sub_schema, child_path)

            # ---- array 专用检查 ----
            if type_req == "array":
                items_schema = schema.get("items")
                if items_schema:
                    for idx, item in enumerate(value):
                        child_path = f"{path}[{idx}]"
                        _check(item, items_schema, child_path)

            # ---- 数值 / 字符串约束 ----
            if isinstance(value, (int, float)):
                if "minimum" in schema and value < schema["minimum"]:
                    _err(f"value {value} < minimum {schema['minimum']}")
                if "maximum" in schema and value > schema["maximum"]:
                    _err(f"value {value} > maximum {schema['maximum']}")

            if isinstance(value, str) and "pattern" in schema:
                import re
                if not re.fullmatch(schema["pattern"], value):
                    _err(f"value '{value}' does not match pattern /{schema['pattern']}/")

            # ---- 枚举 ----
            if "enum" in schema and value not in schema["enum"]:
                _err(f"value {value} not in allowed enum {schema['enum']}")

        _check(self, schema, "<root>")
            
            
CONFIG_SCHEMA = {
    "type": "object",
    "required": ["profile_config", "export_config"],
    "additionalProperties": False,

    "properties": {
        "profile_config": {
            "type": "object",
            "required": [
                "dataset_path", "output_path", "num_samples", "no_bias", "model_config"
            ],
            "additionalProperties": False,

            "properties": {
                "dataset_path": {"type": "string"}, # which dataset to use for profiling
                "output_path":  {"type": "string"}, # where to save the profiling results
                "num_samples":  {"type": "integer", "minimum": 2}, # number of samples to use in dataset to profile
                "no_bias":      {"type": "boolean"}, # if true, we will ignore bias when profiling a linear layer. that is, for a linear layer Wx + b, we will only record the output scale of Wx.

                "model_config": {
                    "type": "object",
                    "required": [
                        "model_type", # currently only support qwen2 and qwen-vl(this is qwen2-vl, not qwen2.5-vl. you can refer to model_interface.py for details)
                        "tokenizer_name", # path to tokenizer
                        "model_name",     # path to model
                    ],
                    "additionalProperties": True,

                    "properties": {
                        "model_type":     {"type": "string"},
                        "tokenizer_name": {"type": "string"},
                        "model_name":     {"type": "string"},
                        "online_rotation": {"type": "boolean"}, # rotate after loading model
                        "random_rotate":   {"type": "boolean"}, # generate random rotation matrix and use it to rotate the model
                        "save_rotation":   {"type": "string"},  # this is the path to save the rotation matrix
                        "R_path": {"type": "string"} # if online_rotation is true, rotation matrix from R_path will be used to rotate the model. random_rotate and  R_path and random_rotate are mutually exclusive
                    }
                }
            }
        },

        "export_config": {
            "type": "object",
            "required": [
                "scale_file", "output_model", "model_config"
            ],
            "additionalProperties": False,

            "properties": {
                "scale_file":        {"type": "string"},
                "output_model":      {"type": "string"},
                "t01m_clip_threshold": {"type": "integer"},
                "quant_bias":        {"type": "boolean"},
                "clip_all":          {"type": "boolean"}, # if true, t01m_clip_threshold will not be effected
                
                "quantize_vit":      {"type": "boolean"}, # if true, we will quantize vit model

                "model_config": {
                    "type": "object",
                    "required": [
                        "model_type",
                        "tokenizer_name",
                        "model_name",
                    ],
                    "additionalProperties": True,

                    "properties": {
                        "model_type":     {"type": "string"},
                        "tokenizer_name": {"type": "string"},
                        "model_name":     {"type": "string"},
                        "online_rotation": {"type": "boolean"},
                        "random_rotate":   {"type": "boolean"},
                        "save_rotation":   {"type": "string"},
                        "R_path": {"type": "string"} # R_path and random_rotate are mutually exclusive
                    }
                }
            }
        }
    }
}

def validate_config(config: ConfigDict):
    if config.profile_config.model_config.online_rotation:
        if not config.profile_config.no_bias:
            raise ValueError("online_rotation requires no_bias to be true")
        
        if config.export_config.quant_bias:
            raise ValueError("quant_bias cannot be true when online_rotation is enabled")
        
    assert config.profile_config.model_config.model_type == config.export_config.model_config.model_type, \
        "model_type in profile_config and export_config must match"
        
    assert config.profile_config.model_config.tokenizer_name == config.export_config.model_config.tokenizer_name, \
        "tokenizer_name in profile_config and export_config must match"
        
    assert config.profile_config.model_config.model_name == config.export_config.model_config.model_name, \
        "model_name in profile_config and export_config must match"

        
if __name__ == "__main__":
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age":  {"type": "integer", "minimum": 0},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "zip":    {"type": "string", "pattern": r"^\d{5}$"}
                },
                "required": ["street", "zip"],
                "additionalProperties": False
            }
        },
        "required": ["name", "address"],
        "additionalProperties": False
    }

    cfg = ConfigDict({
        "name": "Alice",
        "age": 30,
        "address": {
            "street": "Main St",
            "zip": "12345"
        }
    })

    cfg.check_schema(schema)
