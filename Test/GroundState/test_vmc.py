from pytest import approx, raises
import numpy as np

import netket as nk
import netket.variational as vmc

SEED = 214748364


def _setup_vmc():
    L = 4
    g = nk.graph.Hypercube(length=L, n_dim=1)
    hi = nk.hilbert.Spin(s=0.5, graph=g)

    ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(seed=SEED, sigma=0.01)

    ha = nk.operator.Ising(hi, h=1.0)
    sa = nk.sampler.ExactSampler(machine=ma, batch_size=100)
    sa.seed(SEED)
    op = nk.optimizer.Sgd(learning_rate=0.1)

    # Add custom observable
    X = [[0, 1], [1, 0]]
    sx = nk.operator.LocalOperator(hi, [X] * L, [[i] for i in range(8)])

    driver = nk.variational.Vmc(ha, sa, op, 1000)

    return ha, sx, ma, sa, driver


def test_vmc_functions():
    ha, sx, ma, sampler, driver = _setup_vmc()

    driver.advance(200)

    state = ma.to_array()

    exact_dist = np.abs(state) ** 2

    for op, name, tol in (ha, "ha", 1e-6), (sx, "sx", 1e-2):
        print("Testing expectation of op={}".format(name))

        # exact_locs = [vmc.local_value(op, ma, v) for v in ma.hilbert.states()]
        states = np.array(list(ma.hilbert.states()))
        exact_locs = nk.operator.local_values(
            op,
            ma,
            states,
            np.fromiter(
                (ma.log_val(x) for x in states),
                dtype=np.complex128,
                count=states.shape[0],
            ),
        )
        exact_ex = np.sum(exact_dist * exact_locs).real

        data = vmc.compute_samples(
            sampler, n_samples=15000, n_discard=1000, der_logs="centered"
        )

        local_values = nk.operator.local_values(op, ma, data.samples, data.log_values)
        ex = nk.stats.statistics(local_values)
        assert ex.mean.real == approx(np.mean(local_values).real, rel=tol)
        assert ex.mean.real == approx(exact_ex.real, rel=tol)

        ex_hl = vmc.estimate_expectation(op, ma, data, return_gradient=False)
        assert ex_hl.mean == ex.mean
        assert ex_hl.variance == ex.variance
        assert ex_hl.error_of_mean == ex.error_of_mean
        assert ex_hl.R == ex.R

    local_values = nk.operator.local_values(ha, ma, data.samples, data.log_values)
    # ex = vmc.statistics(local_values, n_chains=sampler.batch_size)
    # assert ex.variance == approx(0.0, abs=2e-7)
    grad = vmc.gradient_of_expectation(local_values, data.der_logs)
    assert grad.shape == (ma.n_par,)
    assert np.mean(np.abs(grad) ** 2) == approx(0.0, abs=1e-9)

    _, grad_hl = vmc.estimate_expectation(ha, ma, data, return_gradient=True)
    assert grad_hl.shape == (ma.n_par,)
    assert np.all(grad == grad_hl)

    data_without_logderivs = vmc.compute_samples(
        sampler, n_samples=1, n_discard=1, der_logs=None
    )
    with raises(
        ValueError,
        match=r"`return_gradient=True` passed to `estimate expectation`, but "
        r"`mc_data.der_logs` is not available\. Call `compute_samples\(..., "
        r"der_logs='centered'\)`",
    ):
        vmc.estimate_expectation(op, ma, data_without_logderivs, return_gradient=True)


def test_vmc_use_cholesky_compatibility():
    ha, _, ma, sampler, _ = _setup_vmc()

    op = nk.optimizer.Sgd(learning_rate=0.1)
    with raises(
        ValueError,
        match="Inconsistent options specified: `use_cholesky && sr_lsq_solver != 'LLT'`.",
    ):
        vmc = nk.variational.Vmc(
            ha, sampler, op, 1000, use_cholesky=True, sr_lsq_solver="BDCSVD"
        )
