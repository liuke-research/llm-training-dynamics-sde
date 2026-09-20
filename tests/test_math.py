import numpy as np
import pytest
from scipy.stats import binom
from llm_sde.sde import ou_process,double_well,double_well_density
from llm_sde.markov import stationary_distribution,absolute_spectral_gap,scgf,rate_function,finite_horizon_tail,QuantileChain


def test_ou_mean_variance():
    x=ou_process(seed=123,steps=120000,dt=.1)[10000:]
    assert abs(np.mean(x))<.02
    assert abs(np.var(x)-.125)<.01


def test_ou_reproducible():
    assert np.array_equal(ou_process(seed=1,steps=10),ou_process(seed=1,steps=10))


def test_ou_euler_formula():
    x=ou_process(theta=2,sigma=0,dt=.1,steps=1,x0=1,scheme="euler")
    assert x.tolist()==[1.0,.8]


def test_unstable_ou_euler_rejected():
    with pytest.raises(ValueError):ou_process(theta=2,dt=1,scheme="euler")


def test_double_well_formula_no_noise():
    x=double_well(sigma=0,dt=.01,steps=1,x0=2,scheme="euler")
    assert x[-1]==pytest.approx(1.94)


def test_density_normalized_symmetric():
    x=np.linspace(-4,4,1001);p=double_well_density(x,.7)
    area=np.sum((p[:-1]+p[1:])*np.diff(x)/2)
    assert area==pytest.approx(1) and np.allclose(p,p[::-1])


def test_stationary_and_gap_two_state():
    p=np.array([[.9,.1],[.2,.8]])
    pi=stationary_distribution(p)
    assert np.allclose(pi,[2/3,1/3]) and np.allclose(pi@p,pi)
    assert absolute_spectral_gap(p)==pytest.approx(.3)


def test_nonunique_stationary_rejected():
    with pytest.raises(ValueError):stationary_distribution(np.eye(2))


def test_periodic_gap_zero():
    assert absolute_spectral_gap(np.array([[0,1],[1,0]]))==pytest.approx(0)


@pytest.mark.parametrize("k",[-10,-2,0,2,10])
def test_scgf_bernoulli_closed_form(k):
    p=.3;matrix=np.array([[1-p,p],[1-p,p]])
    assert scgf(matrix,np.array([0,1]),k)==pytest.approx(np.log(1-p+p*np.exp(k)),abs=1e-10)


def test_rate_bernoulli_kl():
    p=.3;a=.6;matrix=np.array([[1-p,p],[1-p,p]])
    actual=rate_function(matrix,np.array([0,1]),a)["rate_grid_lower_approximation"]
    expected=a*np.log(a/p)+(1-a)*np.log((1-a)/(1-p))
    assert abs(actual-expected)<.001


@pytest.mark.parametrize("count",[0,3,7,10])
def test_finite_horizon_exact_binomial(count):
    p=.3;matrix=np.array([[1-p,p],[1-p,p]])
    tail=finite_horizon_tail(matrix,np.array([0,1]),10,count,initial=np.array([1.,0.]))
    assert tail==pytest.approx(binom.sf(count-1,10,p),abs=1e-12)


def test_frozen_bins_no_heldout_refit():
    rng=np.random.default_rng(1)
    chain=QuantileChain.fit(rng.normal(size=1000))
    edges=chain.edges.copy()
    chain.encode(np.array([-100,100]))
    assert np.array_equal(edges,chain.edges)
    assert np.allclose(chain.transition.sum(axis=1),1)


def test_constant_chain_observations_rejected():
    with pytest.raises(ValueError):QuantileChain.fit(np.ones(100))


@pytest.mark.parametrize("matrix",[np.array([[1,1],[0,1]]),np.array([[1,-1],[1,1]]),np.ones((2,3))])
def test_bad_transition_rejected(matrix):
    with pytest.raises(ValueError):scgf(matrix,np.array([0,1]),0)
