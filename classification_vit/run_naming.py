"""Universal run-name resolver. Used identically by train and eval scripts.

A loss flag is considered ACTIVE when its float value > 0; suffix tokens are
appended in the order listed in ``_AUX_FLAGS`` (NOT alphabetical), joined by
'_'. The order is part of the contract — see tests/test_run_naming.py.
"""

# Order matters — keep stable across releases.
# (cli_flag_name, suffix_token, formatter)
_AUX_FLAGS = [
    ('gamma_angular_max',     'angular',  lambda v: f"angular{v:g}"),
    ('eta_proto_max',         'proto',    lambda v: f"proto{v:g}"),
    ('zeta_radvar_max',       'radvar',   lambda v: f"radvar{v:g}"),
    ('xi_betacap_max',        'betacap',  lambda v: f"betacap{v:g}"),
    ('eta_max',               'hhl',      lambda v: f"hhl{v:g}"),
    ('use_cls_depth_residual','clsres',   lambda v: 'clsres'),
]


def resolve_run_name(base_run_name: str, args) -> str:
    """Append active-loss suffix tokens to the base run name.

    Examples:
      base='haa_terminal', flags={eta_proto_max=0.1, zeta_radvar_max=0.05}
        -> 'haa_terminal_proto0.1_radvar0.05'
      base='haa_terminal', flags={} (all zero)
        -> 'haa_terminal'
    """
    suffixes = []
    for flag_name, _, formatter in _AUX_FLAGS:
        val = float(getattr(args, flag_name, 0.0) or 0.0)
        if val > 0:
            suffixes.append(formatter(val))
    if not suffixes:
        return base_run_name
    return base_run_name + '_' + '_'.join(suffixes)
