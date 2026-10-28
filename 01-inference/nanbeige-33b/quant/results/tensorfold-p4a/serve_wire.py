"""Check the approved P2 wire format with a guarded native KV4 server."""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ENGINE = Path('/home/riels/Projects/Personal/Research/agent-training/01-inference/tools/TensorFold-llama')
sys.path.insert(0, str(ENGINE / 'tools/zig'))
from llama_serve_check import guard_command, post, stream


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--need-mib', type=int, default=4096)
    args = p.parse_args()
    assert os.environ.get('TF_LLAMA_KV_BITS') == '4'
    args.out.mkdir(parents=True, exist_ok=True)
    binary = ENGINE / 'zig-out/native/bin/tensorfold-native'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    command = [binary, 'serve', args.model, '--port', str(port), '--name', 'nanbeige-kv4',
               '--context', '1024', '--parallel', '1', '--no-drafts', '--temperature', '0',
               '--prompt-cache-gib', '0', '--keep-warm', '0', '--max-tokens', '512', '--no-update-check']
    guarded = guard_command(args.out, 'server', args.need_mib, command, 120)
    child = None
    with (args.out / 'server.log').open('w') as log:
        guard = subprocess.Popen(guarded, cwd=ENGINE, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if guard.poll() is not None:
                    raise RuntimeError((args.out / 'server.log').read_text())
                children = Path(f'/proc/{guard.pid}/task/{guard.pid}/children').read_text().split()
                for pid in children:
                    exe = Path(f'/proc/{pid}/exe')
                    if exe.exists() and exe.resolve() == binary.resolve():
                        child = int(pid)
                try:
                    with urllib.request.urlopen(url + '/v1/models', timeout=.3) as r:
                        assert r.status == 200
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(.05)
            else:
                raise TimeoutError('native server readiness')
            body = {'model': 'nanbeige-kv4', 'messages': [{'role': 'user', 'content': 'Say hello.'}],
                    'temperature': 0, 'max_tokens': 16}
            first = post(url, body)['response']
            second = post(url, body)['response']
            assert first['choices'] == second['choices']
            events = stream(url, body)
            assembled = {'content': '', 'reasoning_content': ''}
            for event in events:
                for choice in event.get('choices', []):
                    for key in assembled:
                        assembled[key] += choice.get('delta', {}).get(key) or ''
            message = first['choices'][0]['message']
            for key, value in assembled.items():
                assert value == (message.get(key) or ''), (key, value, message)
            try:
                post(url, dict(body, max_tokens=100000))
            except urllib.error.HTTPError as e:
                assert e.code == 400
            else:
                raise AssertionError('context overrun was accepted')
            recovered = post(url, body)['response']
            assert recovered['choices'] == first['choices']
            report = {'state': 'PASS', 'kv_bits': 4, 'layers': 2, 'json_repeat': True,
                      'json_sse_equal': True, 'refusal_status': 400, 'recovery_equal': True,
                      'usage': first['usage']}
            (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report))
        finally:
            if child is not None and Path(f'/proc/{child}/exe').exists():
                assert Path(f'/proc/{child}/exe').resolve() == binary.resolve()
                os.kill(child, signal.SIGTERM)
            guard.wait(timeout=20)
    receipt = json.loads((args.out / 'server-vram.json').read_text())
    assert not receipt['timeout'] and receipt['minimum_free_mib'] >= 2048
    assert receipt['peak_process_vram_mib'] <= args.need_mib


if __name__ == '__main__':
    main()
