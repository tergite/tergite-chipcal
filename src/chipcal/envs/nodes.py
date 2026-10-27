from chipcal.envs.multimode_nodes import run_node as _run

def run_node(theta, k, action, rng, qubits=("q0", "q1"), backend="multimode", noise=None, spectroscopy=None):
    if backend != "multimode":
        raise ValueError("Only multimode is supported")
    return _run(theta, k, action, rng, qubits, noise=noise)
