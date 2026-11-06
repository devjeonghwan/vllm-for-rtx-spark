"""Short-window streaming throughput benchmark.

Use --execute to send requests to the selected vLLM server.
Defaults: four workloads, concurrency 1/16, 20 seconds, two repeats, 512 tokens.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import re
import statistics
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
SPEC = {
    'steps': 'vllm:spec_decode_num_drafts_total',
    'drafted': 'vllm:spec_decode_num_draft_tokens_total',
    'accepted': 'vllm:spec_decode_num_accepted_tokens_total',
    'per_position': 'vllm:spec_decode_num_accepted_tokens_per_pos_total',
}
VALIDATION_PACKAGES = ('jsonschema', 'jsonschema-specifications', 'referencing', 'attrs', 'rpds-py')


@lru_cache(maxsize=64)
def schema_validator(schema_json):
    import jsonschema
    schema = json.loads(schema_json)
    validator_type = jsonschema.validators.validator_for(schema)
    validator_type.check_schema(schema)
    return validator_type(schema)


def schema_validation_preflight(workloads):
    schemas = []
    for workload in workloads:
        response_format = workload.get('response_format')
        if response_format:
            if response_format.get('type') != 'json_schema':
                raise ValueError('This client requires an explicit guided JSON schema')
            raw = json.dumps(response_format['json_schema']['schema'], sort_keys=True)
            validator = schema_validator(raw)  # Import/check before warmup or any timed window.
            schemas.append({'sha256': hashlib.sha256(raw.encode()).hexdigest(),
                            'draft': validator.META_SCHEMA.get('$id')})
    return {'ready': True, 'checked_schemas': schemas,
            'scope': 'Client dependency import and schema checking outside timed windows'}


def validation_dependencies():
    """No imports of native/model packages; usable on either client OS."""
    versions = {}
    for package in VALIDATION_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {'http_and_counting': 'Python standard library only',
            'guided_schema_validation': 'jsonschema >=4 with its declared mandatory dependencies; no format extras needed by this schema',
            'installed_versions': versions}


def guided_output_audit(body, text, finish_reason):
    """Validate only EOS-complete JSON; truncation is a separate incomplete result."""
    result = {'output_ceiling_truncated': finish_reason == 'length',
              'guided_json_requested': bool(body.get('response_format')),
              'guided_json_parse_success': None, 'guided_json_schema_valid': None,
              'guided_validation_failed': False, 'output_quality_pass': None,
              'guided_validation_status': 'not_applicable_no_response_schema'}
    if result['output_ceiling_truncated']:
        result.update(output_quality_pass=False,
                      guided_validation_status='not_validated_output_ceiling_truncation')
        return result
    response_format = body.get('response_format')
    if not response_format:
        return result  # Natural text has no independently proven quality pass.
    if finish_reason != 'stop':
        result.update(guided_validation_failed=True, output_quality_pass=False,
                      guided_validation_status='failed_non_EOS_finish_reason')
        return result
    try:
        value = json.loads(text)
    except ValueError as error:
        result.update(guided_json_parse_success=False, guided_validation_failed=True,
                      output_quality_pass=False, guided_validation_status='failed_JSON_parse_after_EOS',
                      guided_validation_error=str(error))
        return result
    result['guided_json_parse_success'] = True
    if response_format.get('type') != 'json_schema':
        result.update(guided_validation_failed=True, output_quality_pass=False,
                      guided_validation_status='failed_unsupported_response_format')
        return result
    try:
        import jsonschema
        schema = response_format['json_schema']['schema']
        validator = schema_validator(json.dumps(schema, sort_keys=True))
        validator.validate(value)
        result.update(guided_json_schema_valid=True, output_quality_pass=True,
                      guided_validation_status='passed_JSON_schema_after_EOS',
                      guided_schema_draft=validator.META_SCHEMA.get('$id'))
    except ImportError as error:
        result.update(guided_validation_failed=True, output_quality_pass=False,
                      guided_validation_status='failed_client_validation_dependency',
                      guided_validation_error=str(error))
    except (KeyError, TypeError) as error:
        result.update(guided_validation_failed=True, output_quality_pass=False,
                      guided_validation_status='failed_schema_contract', guided_validation_error=str(error))
    except (jsonschema.ValidationError, jsonschema.SchemaError) as error:
        result.update(guided_json_schema_valid=False, guided_validation_failed=True,
                      output_quality_pass=False, guided_validation_status='failed_JSON_schema_after_EOS',
                      guided_validation_error=error.message,
                      guided_validation_path=list(error.path))
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    # Callers own a freshly created directory. Preserve every request separately.
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')


def build_body(workload, prompt_index, model, max_tokens):
    body = {'model': model, 'messages': workload['messages'][prompt_index % len(workload['messages'])],
            'max_tokens': max_tokens, 'temperature': 1.0, 'top_p': 0.95,
            'ignore_eos': False, 'n': 1, 'chat_template_kwargs': {'enable_thinking': False},
            'stream': True, 'stream_options': {'include_usage': True}, 'return_token_ids': True}
    if workload.get('response_format'):
        body['response_format'] = workload['response_format']
    return body


def token_ids(event):
    result, content_without_ids = [], False
    for choice in event.get('choices', []):
        if choice.get('index', 0) != 0:
            raise RuntimeError('Only n=1 is supported')
        ids = choice.get('token_ids')
        if ids is not None:
            if not isinstance(ids, list) or any(type(value) is not int for value in ids):
                raise RuntimeError('Malformed delta token_ids; never infer counts from text')
            result.extend(ids)
        delta = choice.get('delta', {})
        if (delta.get('content') or delta.get('reasoning_content')) and ids is None:
            content_without_ids = True
    return result, content_without_ids


def parse_metrics(raw):
    samples = {}
    for line in raw.splitlines():
        match = re.fullmatch(r'(vllm:[^ {]+)(?:\{([^}]*)\})?\s+(\S+)(?:\s+\S+)?', line)
        if not match:
            continue
        labels = {key: json.loads('"' + value + '"') for key, value in
                  re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', match[2] or '')}
        value = float(match[3])
        if not math.isfinite(value):
            continue
        key = json.dumps([match[1], labels], sort_keys=True, separators=(',', ':'))
        if key in samples:
            raise RuntimeError('Duplicate complete metric labels: ' + key)
        samples[key] = {'name': match[1], 'labels': labels, 'value': value}
    return samples


def get_metrics(base, path):
    with urllib.request.urlopen(base + '/metrics', timeout=30) as response:
        raw = response.read().decode('utf-8')
    path.write_text(raw, encoding='utf-8')
    return parse_metrics(raw)


def metric_delta(before, after, model):
    values = []
    for key, item in after.items():
        if key not in before or not item['name'].endswith(('_total', '_sum', '_count')):
            continue
        if item['labels'].get('model_name', model) != model:
            continue
        difference = item['value'] - before[key]['value']
        if difference < 0:
            raise RuntimeError('Counter reset: ' + key)
        values.append(dict(item, before=before[key]['value'], after=item['value'], delta=difference))
    counters = {}
    labelled = {}
    for name, metric_name in SPEC.items():
        rows = [item for item in values if item['name'] == metric_name]
        labelled[name] = rows
        counters[name] = rows if name == 'per_position' else (sum(item['delta'] for item in rows) if rows else None)
    counters['all_four_families_observed'] = all(labelled.values())
    counters['labelled_samples'] = labelled
    counters['mean_acceptance_length_including_bonus'] = (
        None if not counters['steps'] or counters['accepted'] is None else 1 + counters['accepted'] / counters['steps'])
    return {'samples': values, 'speculative': counters}


def request_one(base, body, directory, start, deadline):
    directory.mkdir(exist_ok=False)
    write_json(directory / 'request.json', body)
    request_start = time.perf_counter()
    row = {'status': 'started', 'request_sha256': sha(directory / 'request.json'),
           'start_seconds_from_window': request_start - start,
           'window_token_ids_count': 0, 'returned_delta_token_ids_count': 0,
           'content_without_token_ids': False, 'received_DONE': False,
           'usage': None, 'finish_reason': None, 'error': None}
    write_json(directory / 'result.json', row)
    content, ids_all = [], []
    try:
        request = urllib.request.Request(base + '/v1/chat/completions',
            data=json.dumps(body, ensure_ascii=False).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with (directory / 'chunks.jsonl').open('x', encoding='utf-8') as journal:
            with urllib.request.urlopen(request, timeout=1200) as response:
                row['http_status'] = response.status
                for raw in response:
                    now = time.perf_counter()
                    line = raw.decode('utf-8').strip()
                    if not line.startswith('data:'):
                        continue
                    payload = line[5:].strip()
                    if payload == '[DONE]':
                        row['received_DONE'] = True
                        break
                    event = json.loads(payload)
                    journal.write(json.dumps({'seconds_from_window': now - start,
                        'within_window': start <= now < deadline, 'event': event}, ensure_ascii=True) + '\n')
                    if event.get('error'):
                        raise RuntimeError(event['error'])
                    delta_ids, missing = token_ids(event)
                    ids_all.extend(delta_ids)
                    row['content_without_token_ids'] |= missing
                    if start <= now < deadline:
                        row['window_token_ids_count'] += len(delta_ids)
                    if event.get('usage') is not None:
                        row['usage'] = event['usage']
                        row['usage_receipt_seconds_from_window'] = now - start
                    for choice in event.get('choices', []):
                        content.append(choice.get('delta', {}).get('content') or '')
                        if choice.get('finish_reason') is not None:
                            row['finish_reason'] = choice['finish_reason']
                    if len(ids_all) % 64 < len(delta_ids):
                        journal.flush()
        row['end_seconds_from_window'] = time.perf_counter() - start
        row['returned_delta_token_ids_count'] = len(ids_all)
        row['completed_within_window'] = row['end_seconds_from_window'] <= deadline - start
        usage = row.get('usage')
        row['token_count_exact_usage_match'] = bool(usage and
            type(usage.get('completion_tokens')) is int and len(ids_all) == usage['completion_tokens'])
        if not row['received_DONE'] or not row['token_count_exact_usage_match'] or row['content_without_token_ids']:
            raise RuntimeError('Incomplete stream or returned token IDs do not exactly match completion usage')
        row.update(guided_output_audit(body, ''.join(content), row['finish_reason']))
        row['status'] = 'complete'
    except BaseException as issue:
        row['status'], row['error'] = 'failed', repr(issue)
        row['traceback'] = traceback.format_exc()
        row['returned_delta_token_ids_count'] = len(ids_all)
        if isinstance(issue, urllib.error.HTTPError):
            (directory / 'http-error-body.bin').write_bytes(issue.read())
    finally:
        row['end_seconds_from_window'] = time.perf_counter() - start
        row['request_wall_seconds'] = time.perf_counter() - request_start
        row['text_characters'] = len(''.join(content))
        (directory / 'text.txt').write_text(''.join(content), encoding='utf-8')
        write_json(directory / 'token_ids.json', ids_all)
        write_json(directory / 'result.json', row)
    return row


def summarize_window(rows, seconds):
    failures = [row for row in rows if row['status'] != 'complete']
    valid = bool(rows) and not failures and all(row.get('token_count_exact_usage_match') and
                                not row['content_without_token_ids'] for row in rows)
    guided_failures = [row for row in rows if row.get('guided_validation_failed')]
    truncated = [row for row in rows if row.get('output_ceiling_truncated')]
    window_valid = valid and not guided_failures
    window_tokens = sum(row['window_token_ids_count'] for row in rows)
    completed = [row for row in rows if row.get('completed_within_window') and row['status'] == 'complete']
    complete_usage = sum(row['usage']['completion_tokens'] for row in completed)
    return {'status': 'completed_observation' if window_valid else 'failed_or_incomplete_observation',
            'window_token_ids_count': window_tokens, 'window_exact_count_available': valid,
            'window_valid': window_valid, 'guided_validation_failures': len(guided_failures),
            'output_ceiling_truncated_requests': len(truncated),
            'output_ceiling_truncation_observed': bool(truncated),
            'benchmark_full_output_eligible': window_valid and not truncated,
            'guided_output_quality_pass': (False if guided_failures or truncated else
                (True if rows and all(row.get('output_quality_pass') is True for row in rows) else None)),
            'diagnostic_exact_window_tokens_per_second': window_tokens / seconds if valid else None,
            'aggregate_window_tokens_per_second': window_tokens / seconds if window_valid else None,
            'completed_within_window_usage_tokens': complete_usage,
            'completed_within_window_usage_tokens_per_second': complete_usage / seconds,
            'completed_within_window_requests': len(completed), 'all_requests_including_tail': len(rows),
            'failed_requests': len(failures), 'partial_counts_preserved': window_tokens,
            'tail_generated_tokens_excluded': sum(row['returned_delta_token_ids_count'] for row in rows) - window_tokens}


def run_window(base, model, workload, concurrency, seconds, max_tokens, directory):
    directory.mkdir(exist_ok=False)
    before = get_metrics(base, directory / 'metrics-before.txt')
    rows, lock, ready, go, state = [], threading.Lock(), threading.Barrier(concurrency + 1), threading.Event(), {}

    def worker(lane):
        ready.wait()
        go.wait()
        count = 0
        while time.perf_counter() < state['deadline']:
            index = lane + count * concurrency
            body = build_body(workload, index, model, max_tokens)
            row = request_one(base, body, directory / ('lane%02d-request%05d' % (lane, count)),
                              state['start'], state['deadline'])
            row.update(lane=lane, lane_request=count, prompt_index=index % len(workload['messages']))
            with lock:
                rows.append(row)
                write_json(directory / 'progress.json', {'elapsed_seconds': time.perf_counter() - state['start'], 'rows': rows})
            if row['status'] != 'complete':
                break  # Preserve first error; do not hammer failing endpoints.
            count += 1

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, lane) for lane in range(concurrency)]
        ready.wait()
        state['start'] = time.perf_counter()
        state['deadline'] = state['start'] + seconds
        go.set()
        # No periodic telemetry. All existing streams drain naturally after the window.
        for future in futures:
            future.result()
    after = get_metrics(base, directory / 'metrics-after-drain.txt')
    result = summarize_window(rows, seconds)
    result.update(rows=rows, measured_seconds=seconds,
                  window_and_tail_wall_seconds=time.perf_counter() - state['start'],
                  metrics_scope='Before window through natural in-flight EOS drain; counters include tail, not strict-window only',
                  metric_delta=metric_delta(before, after, model))
    write_json(directory / 'result.json', result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--base-url', required=True, help='Server root URL, without /v1')
    parser.add_argument('--model', required=True, help='Served model ID')
    parser.add_argument('--output', type=Path, required=True, help='New result directory')
    parser.add_argument('--workloads', type=Path, default=HERE / 'prompts.json')
    parser.add_argument('--cases', default='mixed,code,structured,prose')
    parser.add_argument('--concurrency', default='1,16')
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--warmups', type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='ascii', errors='backslashreplace')
    args = parse_args(argv)
    parsed = urllib.parse.urlparse(args.base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use an HTTP(S) server root URL without credentials, query or fragment')
    args.base_url = args.base_url.rstrip('/')
    workloads_document = json.loads(args.workloads.read_text(encoding='utf-8-sig'))
    for name in args.cases.split(','):
        workload = workloads_document['workloads'][name]
        if not workload['messages'] or not all(isinstance(row, list) and row for row in workload['messages']):
            raise ValueError('Workload must contain nonempty exact messages arrays')
    concurrencies = [int(value) for value in args.concurrency.split(',')]
    if min(concurrencies) < 1 or max(concurrencies) > 16 or not math.isfinite(args.seconds) or args.seconds <= 0 or args.repeats < 1 or args.max_tokens < 1 or args.warmups < 0:
        raise ValueError('Invalid bounded run dimensions')
    plan = {'status': 'source_plan_only', 'HTTP_executed': False,
            'client_dependencies': validation_dependencies(),
            'source_sha256': {str(Path(__file__).resolve()): sha(__file__), str(args.workloads.resolve()): sha(args.workloads)},
            'workload_order': args.cases.split(','), 'concurrency_order': concurrencies,
            'seconds': args.seconds, 'repeats': args.repeats, 'max_tokens_local_ceiling': args.max_tokens,
            'warmups_per_workload_concurrency': args.warmups, 'warmup_max_tokens': 32,
            'seed': 'omitted', 'top_k': 'omitted',
            'window_end_policy': 'Stop replacing at deadline, naturally drain in-flight EOS; exclude tail from window rate',
            'prefix_cache_policy': 'Server configuration; no client salt or cache flush',
            'primary_metric': 'Delta token IDs received in [start,start+seconds), summed across streams / seconds; prefill/refill included',
            'repeat_aggregation': 'Arithmetic mean, with range and median',
            'requests': {name: build_body(workloads_document['workloads'][name], 0, args.model, args.max_tokens)
                         for name in args.cases.split(',')}}
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=True, indent=2))
        return 0
    if args.output is None:
        raise ValueError('--output is required for execution')
    args.output.mkdir(parents=True, exist_ok=False)  # Never overwrite any earlier result, even on failure.
    report = dict(plan, status='started_benchmark', HTTP_executed=True, windows=[], aggregates=[])
    write_json(args.output / 'result.json', report)
    (args.output / 'executed-source.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'workloads.json').write_bytes(args.workloads.read_bytes())
    try:
        report['schema_validation_preflight'] = schema_validation_preflight(
            [workloads_document['workloads'][name] for name in args.cases.split(',')])
        write_json(args.output / 'result.json', report)
        for name in args.cases.split(','):
            workload = workloads_document['workloads'][name]
            for concurrency in concurrencies:
                for index in range(args.warmups):
                    body = build_body(workload, index, args.model, 32)
                    t0 = time.perf_counter()
                    row = request_one(args.base_url, body, args.output / ('%s-c%d-warm%d' % (name, concurrency, index)), t0, t0 + 1200)
                    if row['status'] != 'complete':
                        raise RuntimeError('Warmup failed; see its immutable request/result')
                values = []
                for repeat in range(args.repeats):
                    directory = args.output / ('%s-c%d-repeat%d' % (name, concurrency, repeat))
                    result = run_window(args.base_url, args.model, workload, concurrency, args.seconds, args.max_tokens, directory)
                    report['windows'].append({'workload': name, 'concurrency': concurrency, 'repeat': repeat,
                        'result': str(directory / 'result.json'), 'result_sha256': sha(directory / 'result.json'),
                        **{key: value for key, value in result.items() if key not in ('rows', 'metric_delta')}})
                    write_json(args.output / 'result.json', report)
                    if result['aggregate_window_tokens_per_second'] is None:
                        raise RuntimeError('Incomplete window, preserve counts without token-rate guess')
                    values.append(result['aggregate_window_tokens_per_second'])
                    print(json.dumps({'workload': name, 'concurrency': concurrency, 'repeat': repeat, 'tps': values[-1]}), flush=True)
                report['aggregates'].append({'workload': name, 'concurrency': concurrency, 'runs': values,
                    'arithmetic_mean': statistics.mean(values), 'median': statistics.median(values),
                    'minimum': min(values), 'maximum': max(values)})
        report['all_windows_valid'] = bool(report['windows']) and all(row['window_valid'] for row in report['windows'])
        report['all_windows_full_output_eligible'] = bool(report['windows']) and all(row['benchmark_full_output_eligible'] for row in report['windows'])
        report['any_output_ceiling_truncation'] = any(row['output_ceiling_truncation_observed'] for row in report['windows'])
        report['guided_output_quality_pass'] = (
            False if report['any_output_ceiling_truncation'] else
            (True if report['windows'] and all(row['guided_output_quality_pass'] is True for row in report['windows']) else None))
        report['status'] = ('completed_benchmark' +
            ('_with_output_ceiling_truncation' if report['any_output_ceiling_truncation'] else ''))
    except BaseException as issue:
        report.update(status='failed_benchmark_preserved', error=repr(issue), traceback=traceback.format_exc())
    finally:
        report['source_unchanged'] = all(sha(path) == digest for path, digest in plan['source_sha256'].items())
        write_json(args.output / 'result.json', report)
    return 0 if report['status'].startswith('completed_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
