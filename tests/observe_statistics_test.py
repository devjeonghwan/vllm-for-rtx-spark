"""CPU regression contract for the real depth-observation statistics."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest


def check_statistics(source):
    import torch

    assert not torch.cuda.is_initialized()
    node = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "observe")
    fields = ("st_hist", "st_drafted", "st_accepted")
    cases = [([], []), ([1], [0]), ([4] * 16, [2] * 16),
             ([1, 2, 8, 20], [100, 0, 8, 0]), ([0] * 8, [0] * 8)]
    rounds = 0
    for k in (3, 7, 9):
        for seed in (0, 2**31 + 17):
            state = SimpleNamespace(on=True, active=False, steps=0, K=k,
                pos=torch.arange(1, k + 1, dtype=torch.int32),
                st_hist=torch.full((k + 1,), seed, dtype=torch.int64),
                st_drafted=torch.full((k,), seed, dtype=torch.int64),
                st_accepted=torch.full((k,), seed, dtype=torch.int64))
            callbacks = []
            scope = {"torch": torch, "D": state, "_POLL": 4,
                "_time_step": lambda n: callbacks.append(("time", n)),
                "_cfg": lambda sync: callbacks.append(("cfg", sync)),
                "_stats_write": lambda: callbacks.append(("write", None))}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
            observe = scope["observe"]
            expected = [[seed] * (k + 1), [seed] * k, [seed] * k]
            addresses = [getattr(state, name).data_ptr() for name in fields]

            def run(sampled, rejected):
                for count, rejected_count in zip(sampled, rejected, strict=True):
                    accepted = max(count - 1, 0)
                    drafted = accepted + rejected_count
                    expected[0][min(drafted, k)] += 1
                    for column in range(k):
                        expected[1][column] += int(drafted >= column + 1)
                        expected[2][column] += int(accepted >= column + 1)
                observe(torch.arange(len(sampled), dtype=torch.int64),
                        torch.tensor(sampled, dtype=torch.int32),
                        torch.tensor(rejected, dtype=torch.int32))
                actual = [getattr(state, name).tolist() for name in fields]
                assert actual == expected, (k, seed, sampled, rejected, actual, expected)
                assert addresses == [getattr(state, name).data_ptr() for name in fields]

            for index in range(70):
                run(*cases[index % len(cases)])
            assert state.steps == 70
            assert callbacks.count(("cfg", True)) == callbacks.count(("write", None)) == 1
            assert sum(name == "time" for name, _ in callbacks) == 70
            state.on = False
            prior_callbacks = list(callbacks)
            observe(torch.tensor([0]), torch.tensor([8]), torch.tensor([7]))
            assert state.steps == 70 and callbacks == prior_callbacks
            assert [getattr(state, name).tolist() for name in fields] == expected
            state.on = True
            run([8], [7])
            assert state.steps == 71
            for name in fields:
                getattr(state, name).zero_()
            state.steps = 0
            callbacks.clear()
            expected = [[0] * (k + 1), [0] * k, [0] * k]
            run([1, 8], [0, 0])
            assert state.steps == 1 and callbacks == [("time", 2)]
            rounds += 72
    assert not torch.cuda.is_initialized()
    print(f"CPU observe statistics PASS: {rounds} rounds, CUDA uninitialized")


class ObserveStatisticsTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
    def test_integer_statistics_in_isolated_cpu_process(self):
        source = Path(__file__).resolve().parents[1] / "src/recipes/qwen3_8/helpers/mbx_depth.py"
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        result = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()),
                                 "--worker", str(source)], env=environment,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CPU observe statistics PASS", result.stdout)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        check_statistics(Path(sys.argv[2]))
    else:
        unittest.main()
