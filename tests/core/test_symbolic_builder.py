import numpy as np
import pytest
import sympy as sp

from PyDynSys.core import EuclideanDS, SymbolicSystemBuilder


@pytest.mark.symbolic_builder
@pytest.mark.validation
def test_multi_format_parsing_equivalence():
    t = sp.symbols('t')
    x, y = sp.symbols('x y', cls=sp.Function)
    x, y = x(t), y(t)

    expr_sys = [sp.diff(x, t) - y, sp.diff(y, t) + x]
    str_sys = ["diff(x(t), t) - y(t)", "diff(y(t), t) + x(t)"]

    vf_expr = SymbolicSystemBuilder.build_vector_field(expr_sys, [x, y])
    vf_str = SymbolicSystemBuilder.build_vector_field(str_sys, [x, y])

    state = np.array([1.0, 0.5])
    for time in [0.0, 1.0, np.pi]:
        np.testing.assert_allclose(vf_expr.vector_field(state), vf_str.vector_field(state), rtol=1e-12, atol=1e-12)


@pytest.mark.symbolic_builder
@pytest.mark.validation
def test_first_order_guard_raises():
    t = sp.symbols('t')
    x = sp.symbols('x', cls=sp.Function)
    x = x(t)
    second_order = sp.diff(x, t, 2) + x
    with pytest.raises(ValueError):
        SymbolicSystemBuilder.build_vector_field(second_order, [x])


@pytest.mark.symbolic_builder
@pytest.mark.validation
def test_parameter_substitution_sym_vs_str():
    t = sp.symbols('t')
    x, y = sp.symbols('x y', cls=sp.Function)
    x, y = x(t), y(t)

    a = sp.Symbol('a')
    eqs = [sp.diff(x, t) - a*y, sp.diff(y, t) + x]

    res_sym = SymbolicSystemBuilder.build_vector_field(eqs, [x, y], parameters={a: 2.0})
    res_str = SymbolicSystemBuilder.build_vector_field(eqs, [x, y], parameters={"a": 2.0})

    state = np.array([0.5, -1.0])
    np.testing.assert_allclose(res_sym.vector_field(state), res_str.vector_field(state), rtol=1e-12, atol=1e-12)


@pytest.mark.symbolic_builder
@pytest.mark.validation
def test_autonomy_detection_and_dispatch():
    t = sp.symbols('t')
    x, y = sp.symbols('x y', cls=sp.Function)
    x, y = x(t), y(t)

    # Autonomous
    eqs_auto = [sp.diff(x, t) - y, sp.diff(y, t) + x]
    sys_auto = EuclideanDS.from_symbolic(eqs_auto, [x, y])
    from PyDynSys.core import AutonomousEuclideanDS
    assert isinstance(sys_auto, AutonomousEuclideanDS)

    # Non-autonomous
    eqs_non = [sp.diff(x, t) - y, sp.diff(y, t) + x + sp.sin(t)]
    sys_non = EuclideanDS.from_symbolic(eqs_non, [x, y])
    from PyDynSys.core import NonAutonomousEuclideanDS
    assert isinstance(sys_non, NonAutonomousEuclideanDS)


