import argparse
import os
import sys

# Make ``classification_vit`` importable as a top-level module without
# requiring an __init__.py at the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'classification_vit'))

from run_naming import resolve_run_name


def make_args(**overrides):
    a = argparse.Namespace(
        run_name='base',
        gamma_angular_max=0.0,
        eta_proto_max=0.0,
        zeta_radvar_max=0.0,
        xi_betacap_max=0.0,
        eta_max=0.0,
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def test_no_flags_unchanged():
    a = make_args()
    assert resolve_run_name(a.run_name, a) == 'base'


def test_single_flag():
    a = make_args(eta_proto_max=0.1)
    assert resolve_run_name(a.run_name, a) == 'base_proto0.1'


def test_two_flags_ordered():
    a = make_args(eta_proto_max=0.1, zeta_radvar_max=0.05)
    # Order is fixed by _AUX_FLAGS list, NOT alphabetical.
    # proto comes before radvar in the list.
    assert resolve_run_name(a.run_name, a) == 'base_proto0.1_radvar0.05'


def test_idempotent_train_eval():
    # Train resolves -> eval resolves -> identical
    a_train = make_args(eta_proto_max=0.1, xi_betacap_max=0.3)
    a_eval = make_args(eta_proto_max=0.1, xi_betacap_max=0.3)
    assert (resolve_run_name(a_train.run_name, a_train)
            == resolve_run_name(a_eval.run_name, a_eval))


def test_zero_value_omitted():
    a = make_args(eta_proto_max=0.0, zeta_radvar_max=0.05)
    assert resolve_run_name(a.run_name, a) == 'base_radvar0.05'
