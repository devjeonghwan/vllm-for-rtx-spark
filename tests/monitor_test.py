"""CPU contracts for observer transparency, engine clocks, bounded history and local routes."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch
from support import SOURCE

ROOT = Path(__file__).resolve().parents[1]
_namespace = ModuleType("rtx_monitor_contract")
_namespace.__path__ = [str(ROOT / "src")]
sys.modules["rtx_monitor_contract"] = _namespace
PACKAGE = "rtx_monitor_contract.runtime.monitor"
monitor_api = importlib.import_module(PACKAGE)
state = importlib.import_module(PACKAGE + ".state")


def arguments(**values):
    return NS(**dict(dict(monitor=True, tray=False, api_server_count=1, disable_log_stats=False,
        host="127.0.0.1", port=8000, config="configs/gpu-96gb.yaml", subparser="serve"), **values))


def params(kind="DELTA", n=1):
    return NS(n=n, output_kind=NS(name=kind), temperature=.6, top_p=.95, max_tokens=128)


def output(count=5, metrics_count=5, first=1002., last=1003., finished=False, text="hello"):
    return NS(outputs=[NS(index=0, token_ids=list(range(count)), text=text,
                         finish_reason="stop" if finished else None)], finished=finished,
        prompt="p", prompt_token_ids=list(range(100)), num_cached_tokens=20,
        metrics=NS(num_generation_tokens=metrics_count, arrival_time=1700000000.,
            queued_ts=1000., scheduled_ts=1001., first_token_ts=first, last_token_ts=last,
            first_token_latency=2.5))


class Engine:
    def __init__(self, outputs=()):
        self.outputs = outputs
        self.original_closed = False
        self.logger_manager = NS(stat_loggers=[])
        self.vllm_config = NS(model_config=NS(model="local-model", max_model_len=409600, dtype="bfloat16"),
            cache_config=NS(cache_dtype="fp8", kv_cache_memory_bytes=18000000000, enable_prefix_caching=True),
            speculative_config=NS(num_speculative_tokens=7, draft_model_config=NS(model="draft")),
            additional_config={"rtx_spark":{"draft_head2":True}},
            compilation_config=NS(cudagraph_mode="FULL"), kernel_config=NS(moe_backend="marlin"))

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        try:
            for item in self.outputs:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self.original_closed = True


class ObserverTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_inside_original_await_drains_original(self):
        started = asyncio.Event()
        class WaitingEngine(Engine):
            async def generate(self, *args, **kwargs):
                try:
                    started.set()
                    await asyncio.Event().wait()
                    yield output()
                finally:
                    self.original_closed = True
        engine = WaitingEngine()
        monitor = state.Monitor(engine, arguments())
        async def consume():
            async for _ in engine.generate("p", params(), "cancel-wait"):
                pass
        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(engine.original_closed)
        self.assertEqual(monitor.rows[1]["status"], "cancelled")
        monitor.close()

    async def test_same_output_identity_stats_and_cached_tokens(self):
        outputs = [output(), output(3, 8, last=1004., finished=True)]
        engine = Engine(outputs)
        original = engine.generate
        monitor = state.Monitor(engine, arguments())
        result = [item async for item in engine.generate("prompt", params(), "req")]
        self.assertTrue(all(a is b for a, b in zip(outputs, result, strict=True)))
        row = monitor.rows[1]
        self.assertEqual((row["input"], row["output"], row["cached"]), (100, 8, 20))
        self.assertEqual((row["queue"], row["prefill"], row["decode"], row["ttft"]), (1.,1.,2.,2.5))
        self.assertEqual(row["prefill_rate"], 80.)
        self.assertEqual(row["decode_rate"], 3.5)
        self.assertEqual(monitor.public_row(row,detail=True)["response"], "hellohello")
        self.assertEqual(row["status"], "completed")
        self.assertTrue(engine.original_closed)
        monitor.close(); monitor.close()
        self.assertEqual(engine.generate, original)
        self.assertNotIn(monitor, engine.logger_manager.stat_loggers)

    async def test_close_executes_original_generator_finally(self):
        engine = Engine([output(),output(finished=True)])
        monitor = state.Monitor(engine, arguments())
        stream = engine.generate("p",params(),"cancel")
        await anext(stream)
        await stream.aclose()
        self.assertTrue(engine.original_closed)
        self.assertEqual(monitor.rows[1]["status"],"cancelled")
        monitor.close()

    async def test_underlying_error_is_preserved(self):
        error = RuntimeError("original")
        engine = Engine([output(),error])
        monitor = state.Monitor(engine,arguments())
        with self.assertRaises(RuntimeError) as caught:
            async for _ in engine.generate("p",params(),"failure"):
                pass
        self.assertIs(caught.exception,error)
        self.assertEqual(monitor.rows[1]["status"],"failed")
        self.assertTrue(engine.original_closed)
        monitor.close()

    async def test_monitor_failure_does_not_change_inference(self):
        outputs = [output(),output(finished=True)]
        engine = Engine(outputs)
        monitor = state.Monitor(engine,arguments())
        with patch.object(monitor,"observe",side_effect=ValueError("observer")):
            got=[item async for item in engine.generate("p",params(),"r")]
        self.assertEqual(got,outputs)
        self.assertEqual(monitor.rows[1]["status"],"completed")
        self.assertEqual(monitor.rows[1]["monitor_error"],"ValueError")
        monitor.close()

    async def test_final_only_and_cumulative_do_not_double_count(self):
        for kind in ("FINAL_ONLY","CUMULATIVE"):
            engine=Engine([output(2,2,text="ab"),output(4,4,finished=True,text="abcd")])
            monitor=state.Monitor(engine,arguments())
            async for _ in engine.generate("p",params(kind),"r"):
                pass
            self.assertEqual(monitor.rows[1]["response"],"abcd")
            self.assertEqual(monitor.rows[1]["output"],4)
            monitor.close()

    async def test_single_token_missing_timing_and_multi_sequence_are_not_invented(self):
        for n in (1,2):
            out=output(1,1,first=1002.,last=1002.,finished=True)
            engine=Engine([out]);monitor=state.Monitor(engine,arguments())
            async for _ in engine.generate("p",params(n=n),"r"):
                pass
            row=monitor.rows[1]
            self.assertIsNone(row["decode_rate"])
            if n==2:
                self.assertIsNone(row["output"])
                self.assertIsNone(row["prefill_rate"])
            monitor.close()
        self.assertIsNone(state.interval(100.,0.))
        self.assertIsNone(state.interval(1.,2.))

    async def test_full_text_survives_streaming_cumulative_and_final_only(self):
        prompt = "prompt 한글🙂 " * 2000 + "PROMPT END"
        first = "first 한글🙂 " * 2000
        last = "last text " * 2000 + "RESPONSE END"
        for kind in ("DELTA", "CUMULATIVE", "FINAL_ONLY"):
            outputs = ([output(text=first + last,finished=True)] if kind=="FINAL_ONLY" else
                       [output(text=first),output(text=last if kind=="DELTA" else first+last,finished=True)])
            # Exercise the output.prompt fallback, used when the frontend passes token-only input.
            for item in outputs:
                item.prompt = prompt
            engine=Engine(outputs);monitor=state.Monitor(engine,arguments())
            async for item in engine.generate({"prompt_token_ids":[1]},params(kind),"full"):
                detail=monitor.public_row(monitor.rows[1],detail=True)
                self.assertEqual(detail["prompt"],prompt)
                self.assertEqual(detail["response"],first+last if item.finished else first)
                self.assertNotIn("response",monitor.snapshot()["requests"][0])
                self.assertNotIn("_response",detail)
                self.assertEqual(json.loads(json.dumps(detail))["response"],detail["response"])
            monitor.close()


class StoreTests(unittest.TestCase):
    def test_boundaries_whitelist_scheduler_and_off(self):
        engine=Engine();args=arguments()
        args.api_key="must-not-appear"
        monitor=state.Monitor(engine,args)
        for i in range(120):
            row=monitor.begin(str(i),params(),"x"*20000,{})
            monitor.observe(row,output(text="y"*20000,finished=True))
            monitor.event("INFO","z"*1000)
        self.assertEqual(len(monitor.rows),100)
        self.assertEqual(len(monitor.events),100)
        self.assertEqual(row["prompt"],"x"*20000)
        self.assertEqual(monitor.public_row(row,detail=True)["response"],"y"*20000)
        self.assertNotIn("truncated",row)
        self.assertNotIn(20,monitor.rows)
        snapshot=monitor.snapshot()
        self.assertNotIn('"prompt"',json.dumps(snapshot))
        self.assertNotIn("must-not-appear",json.dumps(snapshot))
        monitor.record(NS(num_running_reqs=2,num_waiting_reqs=3,num_skipped_waiting_reqs=1,kv_cache_usage=.5),None)
        self.assertEqual(monitor.snapshot()["waiting"],4)
        monitor.close()
        fake=NS()
        self.assertIsNone(monitor_api.install(fake,arguments(monitor=False),object()))
        self.assertEqual(vars(fake),{})

    def test_cli_on_off_and_invalid_combinations(self):
        parser=monitor_api.add_cli_args(argparse.ArgumentParser())
        self.assertFalse(parser.parse_args([]).monitor)
        self.assertFalse(parser.parse_args(['--monitor','--no-monitor']).monitor)
        self.assertTrue(parser.parse_args(['--monitor','--tray']).tray)
        for options in ({"tray":True,"monitor":False},{"disable_log_stats":True},
                        {"api_server_count":2},{"headless":True},{"grpc":True}):
            with self.assertRaises(ValueError):
                monitor_api.validate_args(arguments(**options))


class RouteTests(unittest.TestCase):
    def test_routes_lifespan_and_local_boundary(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        web=importlib.import_module(PACKAGE+'.web')
        phases=[]
        @asynccontextmanager
        async def life(app):
            phases.append('enter')
            try:
                yield
            finally:
                phases.append('exit')
        app=FastAPI(lifespan=life)
        engine=Engine()
        original=engine.generate
        monitor=web.install_monitor(app,arguments(),engine)
        with TestClient(app,base_url='http://127.0.0.1:8000',client=('127.0.0.1',30000)) as client:
            self.assertEqual(client.get('/monitor/').status_code,200)
            self.assertEqual(client.get('/monitor/app.js').status_code,200)
            for name, signature in (('vllm.png', b'\x89PNG\r\n\x1a\n'),
                                    ('favicon.ico', b'\x00\x00\x01\x00')):
                response = client.get('/monitor/' + name)
                self.assertEqual(response.status_code,200)
                self.assertTrue(response.content.startswith(signature))
                self.assertEqual(response.headers['cache-control'],'no-store')
                self.assertEqual(client.get('/monitor/' + name,headers={'Origin':'https://evil.example'}).status_code,403)
            self.assertEqual(client.get('/monitor/tray.py').status_code,404)
            self.assertEqual(client.get('/monitor/api/state').json()['model'],'local-model')
            row=monitor.begin('long',params(),"input 한글 "*3000,{})
            monitor.observe(row,output(text="output 🙂 "*3000,finished=True))
            detail=client.get('/monitor/api/requests/1').json()
            self.assertEqual(detail['prompt'],"input 한글 "*3000)
            self.assertEqual(detail['response'],"output 🙂 "*3000)
            self.assertEqual(client.get('/monitor/api/state',headers={'Origin':'https://evil.example'}).status_code,403)
            self.assertEqual(client.get('/monitor/api/state',headers={'Host':'evil.example'}).status_code,403)
            self.assertEqual(client.get('/monitor/api/state',headers={'Sec-Fetch-Site':'cross-site'}).status_code,403)
            self.assertEqual(client.get('/monitor/api/requests/999').status_code,404)
            self.assertEqual(client.post('/monitor/api/stop',json={}).status_code,400)
            self.assertEqual(client.post('/monitor/api/stop',json={'confirm':True}).status_code,503)
        self.assertEqual(phases,['enter','exit'])
        self.assertEqual(engine.generate,original)
        self.assertTrue(monitor.closed)
        with TestClient(app,base_url='http://127.0.0.1',client=('192.0.2.1',30000)) as remote:
            self.assertEqual(remote.get('/monitor/').status_code,403)


@unittest.skipUnless(SOURCE, "Set VLLM_RTX_SPARK_SOURCE_DIR for the composed parser")
class ProfileParserTests(unittest.TestCase):
    def test_selected_yaml_profile_is_retained_without_changing_precedence(self):
        from yaml_false_config_test import load_parser
        import tempfile
        Parser = load_parser(SOURCE)
        parser = Parser()
        commands = parser.add_subparsers(dest="subparser", required=True)
        serve = monitor_api.add_cli_args(commands.add_parser("serve"))
        serve.add_argument("model")
        serve.add_argument("--config")
        parent = ROOT / ".build/monitor-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            profile = Path(directory) / "selected.yaml"
            profile.write_text("monitor: true\ntray: false\n", encoding="utf-8")
            args = parser.parse_args(["serve", "checkpoint", "--config", str(profile)])
            self.assertTrue(args.monitor)
            self.assertFalse(args.tray)
            self.assertEqual(args._rtx_spark_config_file, str(profile))
            off = parser.parse_args(["serve", "checkpoint", "--config", str(profile), "--no-monitor"])
            self.assertFalse(off.monitor)
            plain = parser.parse_args(["serve", "checkpoint", "--monitor"])
            self.assertIsNone(plain._rtx_spark_config_file)


if __name__=='__main__':
    unittest.main()
