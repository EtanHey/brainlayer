"""Measure synthetic queries through the shipped stdio bridge and a scratch server."""
import json
import os
import subprocess
import sys

socket_path, root, receipt = sys.argv[1:]
assert socket_path != '/tmp/brainbar.sock'
env = dict(os.environ, PYTHONPATH=root + '/src', BRAINLAYER_MCP_SOCKET=socket_path, BRAINLAYER_FORBID_BRAINBAR_SOCKET="1")
bridge = subprocess.Popen([sys.executable, '-m', 'brainlayer.mcp_stdio_bridge'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)


def request(method, params=None):
    payload = json.dumps(dict(jsonrpc='2.0', id=1, method=method, params=params or {})).encode()
    bridge.stdin.write(b'Content-Length: ' + str(len(payload)).encode() + b'\r\n\r\n' + payload)
    bridge.stdin.flush()
    headers = {}
    while True:
        line = bridge.stdout.readline()
        if line == b'\r\n':
            break
        if not line:
            raise RuntimeError('bridge EOF')
        key, value = line.decode().split(':', 1)
        headers[key.lower()] = value.strip()
    raw = bridge.stdout.read(int(headers['content-length']))
    response = json.loads(raw)
    assert 'error' not in response, response
    assert not response.get('result', {}).get('isError'), response
    return raw, response['result']


try:
    request('initialize', {'protocolVersion': '2024-11-05', 'capabilities': {},
                           'clientInfo': {'name': 'diet-fixture-client', 'version': '1'}})
    request('tools/call', {'name': 'expand_palette', 'arguments': {}})
    rows = []
    for index in range(10):
        query = 'DietEntity' + str(index)
        calls = [
            ('1 compact IDs', 'brain_search', {'query': query, 'project': 'fixture'}),
            ('2 KG default', 'brain_search', {'query': query}),
            ('3 full framing', 'brain_search', {'query': query, 'detail': 'full'}),
            ('4 empty entity', 'brain_entity', {'query': 'EmptyEntity' + str(index)}),
            ('4 empty person', 'brain_get_person', {'name': 'EmptyPerson' + str(index)}),
            ('5 context cap', 'brain_recall', {'mode': 'context', 'session_id': query}),
            ('5 injections cap', 'brain_recall', {'mode': 'injections', 'session_id': query}),
        ]
        for item, tool, arguments in calls:
            raw, result = request('tools/call', {'name': tool, 'arguments': arguments})
            text = '\n'.join(c.get('text', '') for c in result['content'])
            rows.append(dict(item=item, query=query, response_bytes=len(raw), text_bytes=len(text.encode())))
    raw, result = request('tools/list')
    with open(receipt, 'w') as output:
        json.dump(dict(rows=rows, tools_list_bytes=len(raw)), output, indent=2)
finally:
    bridge.terminate()
    bridge.wait(timeout=5)
