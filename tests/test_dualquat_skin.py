import pytest
import graphics.dualquat_skin as mod

CASES = [
    ("normalize_weights_runtime", (1,)),
    ("guard_bone_index", (1,)),
    ("double_buffer_rebuild", (1,)),
    ("continuous_blend_weight", (1,)),
    ("fixed_bone_order", (1,)),
    ("validate_export", (1,)),
    ("load_mesh", (1,)),
    ("rebuild", (1,)),
    ("skin", (1,)),
    ("draw", (1,)),
    ("validate_skin", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
