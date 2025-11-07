from tensorflow.keras import layers, models, Input
from typing import Dict, Any, Tuple

from base.load_config import load_config
from machine_learning.custom_layers import CUSTOM_LAYER_REGISTRY



def build_and_save_model_from_config(config_path: str, compile_model: bool = True):
    model_config = load_config(config_path, module_name="model")
    if not model_config:
        raise ValueError(f"Model config is invalid or empty: {config_path}")

    model_init_config = model_config.get("model_init_config", {})
    layers_param = model_init_config.get("layers_param", [])
    if not layers_param:
        raise ValueError(f"'layers_param' is missing from the model config: {config_path}")

    try:
        input_shape = input_shape_parse(config=model_init_config)
    except Exception as e:
        raise ValueError(f"Could not parse input shape: {e}")

    inputs = Input(shape=input_shape)
    output = inputs
    for layer_info in layers_param:
        layer_name = layer_info.get('layer_name')
        layer_kwargs = layer_info.get('layer_kwargs', {})
        LayerClass = get_layer_class(layer_name)
        output = LayerClass(**layer_kwargs)(output)

    model_name = model_config.get("model_name", "sequential_model_from_yml")
    model = models.Model(inputs=inputs, outputs=output, name=model_name)
    model.summary()

    if compile_model:
        compile_param = model_init_config.get("compile_param", {})
        if compile_param:
            model.compile(**compile_param)

    return model


def get_layer_class(layer_name: str):
    if layer_name in CUSTOM_LAYER_REGISTRY:
        return CUSTOM_LAYER_REGISTRY[layer_name]
    try:
        return getattr(layers, layer_name)
    except AttributeError:
        raise ValueError(f"Could not find layer named '{layer_name}' in Keras or custom registry.")


def input_shape_parse(config: Dict[str, Any]) -> Tuple[int, ...]:
    dims = []
    len_keys = sorted([k for k in config if k.startswith('input_len_')])
    for k in len_keys:
        dims.append(config[k])
    if 'input_channels' in config:
        dims.append(config['input_channels'])

    if not dims:
        raise ValueError("Invalid input shape config: missing 'input_len_x' or 'input_channels'.")

    return tuple(dims)
