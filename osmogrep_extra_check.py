# eval-only extra checks for the fork PR (not for upstream)
import pickle, sys, traceback
import numpy as np
from tinygrad import Tensor, TinyJit, Variable

def case_pickle_returns_current_input():
  @TinyJit
  def f(a):
    a.assign(a + 1)
    return a
  for i in range(3): f(Tensor([float(i)] * 3).contiguous().realize())
  g = pickle.loads(pickle.dumps(f))
  for i in range(10, 14):
    a = Tensor([float(i)] * 3).contiguous().realize()
    np.testing.assert_equal(g(a).numpy(), [i + 1] * 3, err_msg=f"pickled call {i}")
    np.testing.assert_equal(a.numpy(), [i + 1] * 3)

def case_aliased_input_returned():
  # feed a jit output back as input (aliased path with staging copy), function also returns its input
  @TinyJit
  def f(x): return x, (x + 1).realize()
  for i in range(3): f(Tensor([float(i)] * 2).contiguous().realize())
  out = f(Tensor([100.0] * 2).contiguous().realize())[1]
  for _ in range(3):
    v = out.numpy().copy()
    r, out = f(out)
    np.testing.assert_equal(r.numpy(), v, err_msg="returned input wrong on aliased call")
    np.testing.assert_equal(out.numpy(), v + 1, err_msg="output wrong on aliased call")

def case_symbolic_view_of_input():
  @TinyJit
  def f(a, n):
    a.assign(a + 1).realize()
    return a[:n]
  for i in range(1, 7):
    a = (Tensor.arange(8).float() + i * 100).clone().realize()
    out = f(a, Variable("n", 1, 8).bind(i))
    np.testing.assert_equal(out[:i].numpy(), np.arange(i, dtype=np.float32) + i * 100 + 1)

def case_return_input_view_and_output():
  @TinyJit
  def f(x): return x[1:3], (x * 2).realize()
  for i in range(6):
    x = (Tensor.arange(4).float() + i * 100).clone().realize()
    v, y = f(x)
    exp = np.arange(4, dtype=np.float32) + i * 100
    np.testing.assert_equal(y.numpy(), exp * 2, err_msg=f"call {i}: output")
    np.testing.assert_equal(v.numpy(), exp[1:3], err_msg=f"call {i}: returned view is not of this call's input")

def case_earlier_inputs_untouched():
  @TinyJit
  def f(a): return a, (a + 1).realize()
  kept = []
  for i in range(6):
    a = Tensor([float(i)] * 2).contiguous().realize()
    kept.append(a)
    r, b = f(a)
    np.testing.assert_equal(r.numpy(), [i] * 2, err_msg=f"call {i}: returned input")
    np.testing.assert_equal(b.numpy(), [i + 1] * 2)
  for i, a in enumerate(kept): np.testing.assert_equal(a.numpy(), [i] * 2, err_msg=f"input from call {i} was rebound")

if __name__ == "__main__":
  failed = []
  for name, fn in list(globals().items()):
    if not name.startswith("case_"): continue
    try: fn(); print("PASS", name)
    except Exception: failed.append(name); print("FAIL", name); traceback.print_exc(limit=2)
  sys.exit(1 if failed else 0)
