# OsmoGrep generated reproducer (not an upstream test).
# Behavior reported upstream in tinygrad PR #19205 "fix jit returning its inputs":
# a TinyJit function that returns (a view of) its input must return the CURRENT call's input, not the capture-time one.
# Calls: cnt 0 = ignore, 1 = capture, 2+ = exec, so >= 5 calls reach the exec path.
import sys, traceback
import numpy as np
from tinygrad import Tensor, TinyJit, Variable

def case_assign_return_input():
  @TinyJit
  def f(a):
    a.assign(a + 1)
    return a
  for i in range(6):
    a = Tensor([float(i * 10)] * 3).contiguous().realize()
    out = f(a)
    np.testing.assert_equal(a.numpy(), [i * 10 + 1] * 3, err_msg=f"call {i}: input not updated")
    np.testing.assert_equal(out.numpy(), [i * 10 + 1] * 3, err_msg=f"call {i}: returned tensor is not this call's input")

def case_return_input_view_and_output():
  @TinyJit
  def f(x): return x[1:3], (x * 2).realize()
  for i in range(6):
    x = (Tensor.arange(4).float() + i * 100).contiguous().realize()
    v, y = f(x)
    exp = np.arange(4, dtype=np.float32) + i * 100
    np.testing.assert_equal(y.numpy(), exp * 2, err_msg=f"call {i}: computed output wrong")
    np.testing.assert_equal(v.numpy(), exp[1:3], err_msg=f"call {i}: returned view is not of this call's input")

def case_earlier_inputs_untouched():
  # rebinding returned tensors must not silently change tensors the caller passed on earlier calls
  @TinyJit
  def f(a): return a
  kept = []
  for i in range(6):
    a = Tensor([float(i)] * 2).contiguous().realize()
    kept.append(a)
    np.testing.assert_equal(f(a).numpy(), [i] * 2, err_msg=f"call {i}: returned tensor is not this call's input")
  for i, a in enumerate(kept): np.testing.assert_equal(a.numpy(), [i] * 2, err_msg=f"input from call {i} was rebound")

def case_symbolic_view_of_input():
  # returned tensor is both input-derived and symbolic
  @TinyJit
  def f(a, n):
    a.assign(a + 1).realize()
    return a[:n]
  for i in range(1, 7):
    a = (Tensor.arange(8).float() + i * 100).contiguous().realize()
    out = f(a, Variable("n", 1, 8).bind(i))
    np.testing.assert_equal(out[:i].numpy(), np.arange(i, dtype=np.float32) + i * 100 + 1, err_msg=f"call {i}: symbolic view wrong")

def case_global_return_unchanged():
  # returning a non-input tensor that the JIT mutates keeps working (mirrors test_updated_not_freed)
  x = Tensor([1.0]).contiguous().realize()
  @TinyJit
  def f(y):
    x.assign(x + y)
    return x
  for _ in range(5): out = f(Tensor([1.0]).contiguous().realize())
  np.testing.assert_equal(x.numpy(), [6.0])
  np.testing.assert_equal(out.numpy(), [6.0])

if __name__ == "__main__":
  failed = []
  for name, fn in list(globals().items()):
    if not name.startswith("case_"): continue
    try:
      fn(); print("PASS", name)
    except Exception:
      failed.append(name); print("FAIL", name); traceback.print_exc(limit=1)
  print(f"{len(failed)} failed")
  sys.exit(1 if failed else 0)
