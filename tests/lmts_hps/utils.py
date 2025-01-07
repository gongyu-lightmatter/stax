def collect_device_params_for_proc_var(model_params_for_proc_var: dict, device_name: str):
    collected = {}
    for wavelength, params_at_wavelength in model_params_for_proc_var.items():
        collected[wavelength] = {} # Initialize empty dict for each wavelength
        for key, value in params_at_wavelength.items():
            if device_name not in key:
                continue

            collected[wavelength].update({key: value})

    return collected


def _collect_helper(model_params_for_proc_var: dict, device_name: str):
    found_value = None
    for key, value in model_params_for_proc_var.items():
        if device_name in key:
            found_value = value
            break
        elif isinstance(value, dict):
            found_value = _collect_helper(value, device_name)
            if found_value is not None:
                break

    return found_value


def collect_leaf_params_for_proc_var(model_params_for_proc_var: dict, param_name: str):
    collected = {}
    for wavelength, params_at_wavelength in model_params_for_proc_var.items():
        collected[wavelength] = {} # Initialize empty dict for each wavelength
        for key, value in params_at_wavelength.items():
            if param_name in key:
                collected[wavelength] = value
            elif isinstance(value, dict):
                collected[wavelength] = _collect_helper(value, param_name)

        assert collected[
            wavelength] is not None, f"Could not find {param_name} in {model_params_for_proc_var}"

    return collected
