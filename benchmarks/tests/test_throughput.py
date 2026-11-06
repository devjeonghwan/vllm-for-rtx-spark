"""Offline client tests: fake SSE, clocks and requests; no model or HTTP server."""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
THROUGHPUT = HERE.parent / "throughput"
SPEC = importlib.util.spec_from_file_location("benchmark_client", THROUGHPUT / "run.py")
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)
WORKLOADS = json.loads((THROUGHPUT / "prompts.json").read_text(encoding="utf-8"))["workloads"]


@contextmanager
def temporary_directory():
    parent = HERE.parents[1] / ".build/benchmark-client-tests"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        yield Path(directory)


def complete_row(window=4, returned=6, completed=False):
    return {"status": "complete", "window_token_ids_count": window,
            "returned_delta_token_ids_count": returned, "token_count_exact_usage_match": True,
            "content_without_token_ids": False, "completed_within_window": completed,
            "usage": {"completion_tokens": returned}, "guided_validation_failed": False,
            "output_ceiling_truncated": False, "output_quality_pass": None}


def sse_response(events, done=True):
    lines = ["data: " + json.dumps(event) + "\n\n" for event in events]
    if done:
        lines.append("data: [DONE]\n\n")
    response = io.BytesIO("".join(lines).encode("utf-8"))
    response.status = 200
    return response


class BenchmarkClientTests(unittest.TestCase):
    def test_request_contract_and_mixed_prompt_cycle(self):
        mixed = WORKLOADS["mixed"]
        self.assertEqual(len(mixed["messages"]), 8)
        for index in range(32):
            body = CLIENT.build_body(mixed, index, "served/model", 512)
            self.assertEqual(body, {
                "model": "served/model", "messages": mixed["messages"][index % 8],
                "max_tokens": 512, "temperature": 1.0, "top_p": 0.95, "ignore_eos": False,
                "n": 1, "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
                "stream_options": {"include_usage": True}, "return_token_ids": True,
            })
        body = CLIENT.build_body(WORKLOADS["structured"], 0, "served/model", 32)
        self.assertEqual(body["response_format"], WORKLOADS["structured"]["response_format"])
        self.assertEqual(body["max_tokens"], 32)

    def test_token_ids_are_required_and_never_inferred_from_text(self):
        self.assertEqual(CLIENT.token_ids({"choices": [{"delta": {"content": "many words"}, "token_ids": [1]}]}), ([1], False))
        self.assertEqual(CLIENT.token_ids({"choices": [{"delta": {"content": "x"}}]}), ([], True))
        for choice in ({"token_ids": [True]}, {"token_ids": "12"}, {"index": 1, "token_ids": [1]}):
            with self.subTest(choice=choice), self.assertRaises(RuntimeError):
                CLIENT.token_ids({"choices": [choice]})

    def test_sse_window_includes_start_excludes_deadline_and_drains_tail(self):
        events = [
            {"choices": [{"token_ids": [1, 2], "delta": {"content": "before"}}]},
            {"choices": [{"token_ids": [3], "delta": {"content": "at start"}}]},
            {"choices": [{"token_ids": [4, 5, 6], "delta": {"content": "inside"}}]},
            {"choices": [{"token_ids": [7, 8, 9, 10], "finish_reason": "length"}]},
            {"choices": [], "usage": {"completion_tokens": 10}},
        ]
        # Blank SSE separators are read too; each shares its event's timestamp.
        clocks = [98, 99, 99, 100, 100, 119, 119, 120, 120, 121, 121, 122, 123, 124, 125]
        with temporary_directory() as directory, \
                patch.object(CLIENT.urllib.request, "urlopen", return_value=sse_response(events)) as opening, \
                patch.object(CLIENT.time, "perf_counter", side_effect=clocks):
            body = CLIENT.build_body(WORKLOADS["code"], 0, "served/model", 512)
            row = CLIENT.request_one("https://server.example", body, directory / "request", 100, 120)
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["window_token_ids_count"], 4)
            self.assertEqual(row["returned_delta_token_ids_count"], 10)
            self.assertTrue(row["received_DONE"])
            self.assertTrue(row["token_count_exact_usage_match"])
            self.assertFalse(row["completed_within_window"])
            self.assertTrue(row["output_ceiling_truncated"])
            request = opening.call_args.args[0]
            self.assertEqual(request.full_url, "https://server.example/v1/chat/completions")
            self.assertEqual(json.loads(request.data), body)
            summary = CLIENT.summarize_window([row], 20)
            self.assertEqual(summary["aggregate_window_tokens_per_second"], 0.2)
            self.assertEqual(summary["tail_generated_tokens_excluded"], 6)
            self.assertFalse(summary["benchmark_full_output_eligible"])

    def test_incomplete_or_unverifiable_stream_invalidates_rate(self):
        examples = [
            ([{"choices": [{"delta": {"content": "x"}}]}, {"usage": {"completion_tokens": 1}}], True),
            ([{"choices": [{"token_ids": [1]}]}, {"usage": {"completion_tokens": 2}}], True),
            ([{"choices": [{"token_ids": [1]}]}, {"usage": {"completion_tokens": 1}}], False),
        ]
        for events, done in examples:
            with self.subTest(events=events, done=done), temporary_directory() as directory, \
                    patch.object(CLIENT.urllib.request, "urlopen", return_value=sse_response(events, done)):
                row = CLIENT.request_one("http://server.example", {}, directory / "request", 0, float("inf"))
                self.assertEqual(row["status"], "failed")
                summary = CLIENT.summarize_window([row], 20)
                self.assertIsNone(summary["aggregate_window_tokens_per_second"])
                self.assertFalse(summary["window_valid"])
                self.assertEqual(summary["failed_requests"], 1)

    def test_guided_schema_eos_validation_and_truncation_are_distinct(self):
        workload = WORKLOADS["structured"]
        body = CLIENT.build_body(workload, 0, "served/model", 512)
        entry = {"id": 1, "name": "Tent", "description": "A useful tent.",
                 "tags": ["a", "b", "c"], "price": 12.5, "in_stock": True}
        valid = json.dumps({"title": "Catalog", "entries": [entry] * 3})
        CLIENT.schema_validation_preflight([workload])
        self.assertTrue(CLIENT.guided_output_audit(body, valid, "stop")["guided_json_schema_valid"])
        for text in ("not JSON", '{"title":"Catalog","entries":[]}'):
            with self.subTest(text=text):
                self.assertTrue(CLIENT.guided_output_audit(body, text, "stop")["guided_validation_failed"])
        truncated = CLIENT.guided_output_audit(body, "{", "length")
        self.assertTrue(truncated["output_ceiling_truncated"])
        self.assertFalse(truncated["guided_validation_failed"])
        self.assertIsNone(truncated["guided_json_schema_valid"])
        self.assertFalse(truncated["output_quality_pass"])
        self.assertTrue(CLIENT.guided_output_audit(body, valid, "tool_calls")["guided_validation_failed"])

    def test_window_aggregate_uses_received_ids_and_marks_guided_failure(self):
        rows = [complete_row(4, 10), complete_row(6, 6, completed=True)]
        summary = CLIENT.summarize_window(rows, 20)
        self.assertEqual(summary["aggregate_window_tokens_per_second"], 0.5)
        self.assertEqual(summary["completed_within_window_usage_tokens"], 6)
        self.assertEqual(summary["tail_generated_tokens_excluded"], 6)
        rows[0]["guided_validation_failed"] = True
        summary = CLIENT.summarize_window(rows, 20)
        self.assertTrue(summary["window_exact_count_available"])
        self.assertFalse(summary["window_valid"])
        self.assertIsNone(summary["aggregate_window_tokens_per_second"])
        self.assertEqual(summary["diagnostic_exact_window_tokens_per_second"], 0.5)

    def test_closed_loop_lanes_cycle_mixed_prompts_and_stop_at_deadline(self):
        for concurrency, per_lane in ((1, 8), (16, 1)):
            counts = {}
            main_thread = threading.current_thread()
            main_clock = iter((100, 121))

            def clock():
                if threading.current_thread() is main_thread:
                    return next(main_clock)
                return 100 if counts.get(threading.get_ident(), 0) < per_lane else 121

            def request(base, body, directory, start, deadline):
                identity = threading.get_ident()
                counts[identity] = counts.get(identity, 0) + 1
                self.assertEqual((start, deadline), (100, 120))
                return complete_row()

            with self.subTest(concurrency=concurrency), temporary_directory() as directory, \
                    patch.object(CLIENT, "request_one", side_effect=request), \
                    patch.object(CLIENT, "get_metrics", return_value={}), \
                    patch.object(CLIENT.time, "perf_counter", side_effect=clock):
                result = CLIENT.run_window("http://server.example", "served/model", WORKLOADS["mixed"],
                                           concurrency, 20, 512, directory / "window")
                self.assertEqual(len(result["rows"]), concurrency * per_lane)
                self.assertEqual(sorted(row["prompt_index"] for row in result["rows"]),
                                 sorted(index % 8 for index in range(concurrency * per_lane)))
                self.assertEqual(result["measured_seconds"], 20)

    def test_metrics_are_labelled_model_filtered_and_counter_resets_fail(self):
        before = CLIENT.parse_metrics('vllm:spec_decode_num_drafts_total{model_name="served/model",worker="0"} 2\n'
                                      'vllm:spec_decode_num_drafts_total{model_name="other"} 100\n')
        after = CLIENT.parse_metrics('vllm:spec_decode_num_drafts_total{model_name="served/model",worker="0"} 5\n'
                                     'vllm:spec_decode_num_drafts_total{model_name="other"} 200\n')
        delta = CLIENT.metric_delta(before, after, "served/model")
        self.assertEqual(delta["speculative"]["steps"], 3)
        self.assertIsNone(delta["speculative"]["accepted"])
        self.assertIsNone(CLIENT.metric_delta({}, {}, "served/model")["speculative"]["steps"])
        with self.assertRaises(RuntimeError):
            CLIENT.metric_delta(after, before, "served/model")

    def test_plan_is_portable_and_sends_no_http_or_output_files(self):
        with temporary_directory() as directory, redirect_stdout(io.StringIO()) as output, \
                patch.object(CLIENT.urllib.request, "urlopen", side_effect=AssertionError("No HTTP in plan")):
            destination = directory / "not-created"
            self.assertEqual(CLIENT.main(["--base-url", "https://server.example", "--model", "served/model",
                                          "--output", str(destination)]), 0)
            plan = json.loads(output.getvalue())
            self.assertFalse(plan["HTTP_executed"])
            self.assertEqual(plan["workload_order"], ["mixed", "code", "structured", "prose"])
            self.assertEqual(plan["concurrency_order"], [1, 16])
            self.assertEqual((plan["seconds"], plan["repeats"], plan["max_tokens_local_ceiling"]), (20, 2, 512))
            self.assertEqual((plan["warmups_per_workload_concurrency"], plan["warmup_max_tokens"]), (1, 32))
            self.assertFalse(destination.exists())

    def test_execute_preserves_cell_order_one_warmup_and_two_run_mean(self):
        observed = []

        def warmup(base, body, directory, start, deadline):
            observed.append(("warmup", directory.name, body["max_tokens"]))
            return {"status": "complete"}

        def window(base, model, workload, concurrency, seconds, maximum, directory):
            observed.append(("window", directory.name, maximum))
            directory.mkdir()
            result = {"aggregate_window_tokens_per_second": 10 if directory.name.endswith("repeat0") else 20,
                      "window_valid": True, "benchmark_full_output_eligible": False,
                      "output_ceiling_truncation_observed": True, "guided_output_quality_pass": False}
            CLIENT.write_json(directory / "result.json", result)
            return result

        with temporary_directory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(CLIENT, "request_one", side_effect=warmup), \
                patch.object(CLIENT, "run_window", side_effect=window), \
                patch.object(CLIENT.urllib.request, "urlopen", side_effect=AssertionError("No real HTTP")):
            destination = directory / "result"
            self.assertEqual(CLIENT.main(["--base-url", "http://server.example", "--model", "served/model",
                                          "--output", str(destination), "--execute"]), 0)
            expected = []
            for name in ("mixed", "code", "structured", "prose"):
                for concurrency in (1, 16):
                    expected.extend([("warmup", f"{name}-c{concurrency}-warm0", 32),
                                     ("window", f"{name}-c{concurrency}-repeat0", 512),
                                     ("window", f"{name}-c{concurrency}-repeat1", 512)])
            self.assertEqual(observed, expected)
            report = json.loads((destination / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["windows"]), 16)
            self.assertTrue(all(row["arithmetic_mean"] == 15 for row in report["aggregates"]))
            self.assertTrue(report["source_unchanged"])
            self.assertTrue(report["any_output_ceiling_truncation"])
            self.assertFalse(report["all_windows_full_output_eligible"])

    def test_existing_output_is_preserved_before_requests(self):
        with temporary_directory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(CLIENT.urllib.request, "urlopen", side_effect=AssertionError("No HTTP")):
            sentinel = directory / "keep.txt"
            sentinel.write_text("existing result", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                CLIENT.main(["--base-url", "http://server.example", "--model", "served/model",
                             "--output", str(directory), "--execute"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing result")

    def test_server_model_and_output_must_be_explicit(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            CLIENT.parse_args([])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
