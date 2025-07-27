def _infer_quantization_config(quant_config: dict) -> dict | None:
    """Construct linear_config dictionary carrying FP8 configuration for FMS.

    There's many quantization packages compatible with HF
    We initially focus on llm-compressor as it is the one used in FMS-MO

    llm-compressor saves its checkpoints with quant_method = compressed-tensors
    quantization_status tells us whether the model has already been quantized
    We only support loading already quantized models (compressed status)
    """

    if (
        quant_config["quant_method"] == "compressed-tensors"
        and quant_config["quantization_status"] == "compressed"
    ):
        # FP8 quantization will have FP8 weights
        # We assume a single quantization group (group_0), to follow fms-mo checkpoints
        # num_bits and type tells us "float" with "8" bits, aka FP8
        if (
            quant_config["config_groups"]["group_0"]["weights"]["type"] == "float"
            and quant_config["config_groups"]["group_0"]["weights"]["num_bits"] == 8
        ):
            # First, import required FP8 linear classes from fms-mo
            # Local
            import fms_mo.aiu_addons.fp8.fp8_adapter  # pylint: disable=unused-import
            import fms_mo.aiu_addons.fp8.fp8_linear  # pylint: disable=unused-import

            # This is used by get_linear to decide whether a linear layer
            # will be quantized or not inside the model
            def fp8_linear_type(name: str) -> str:
                # We need to translate HF names to FMS names
                translations = {
                    "lm_head": "head",
                }
                for ignored_layer in quant_config["ignore"]:
                    assert isinstance(ignored_layer, str)
                    fms_ign_layer = translations.get(ignored_layer, ignored_layer)
                    if name in fms_ign_layer:
                        return "torch_linear"
                for pattern in quant_config["config_groups"]["group_0"]["targets"]:
                    # Special case from llm-compressor that covers all linear layers
                    # not in the ignore pattern
                    assert isinstance(pattern, str)
                    if pattern == "Linear":
                        return "fp8"
                    if name in translations.get(pattern, pattern):
                        return "fp8"
                return "torch_linear"

            return {
                "linear_type": fp8_linear_type,
                "input_activations": quant_config["config_groups"]["group_0"][
                    "input_activations"
                ],
                "output_activations": quant_config["config_groups"]["group_0"][
                    "output_activations"
                ],
                "weights": quant_config["config_groups"]["group_0"]["weights"],
            }
    return None
