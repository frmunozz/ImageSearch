

def validate_keys(kwargs, good_kwargs):
    """
    validate keyword arguments,

    checking if there is any bad keyword,
    and defining with their default value all the non-defined keywords.

    :param kwargs: the input keyword dictionary, with only the defined keywords.
    :param good_kwargs: the valid keywords with their default values.
    :return: the output keywords dictionary, with all the keywords.
    """
    good_keys = set(good_kwargs)
    bad_keys = set(kwargs) - good_keys
    if bad_keys:
        bad_keys = ", ".join(bad_keys)
        raise KeyError("Unknown parameters: {}".format(bad_keys))

    new_kwargs = {}
    for k in good_keys:
        new_kwargs[k.rstrip("_")] = kwargs.get(k, good_kwargs.get(k))
    return new_kwargs