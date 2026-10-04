"""Fresh-process help/list/import gates with internet sockets blocked."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GUARD = '''
import runpy, socket, sys
def guarded(original):
    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('No-download command attempted a network connection.')
        return original(sock, address)
    return connect
socket.socket.connect = guarded(socket.socket.connect)
socket.socket.connect_ex = guarded(socket.socket.connect_ex)
'''


def main() -> int:
    for args in (["--help"], ["--list-models"], ["--list-kokoro-voices"]):
        script = GUARD + f"sys.argv = ['Libro-tts.py', *{args!r}]; runpy.run_path('Libro-tts.py', run_name='__main__')"
        subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT, check=True, timeout=20)
    script = GUARD + '''
import libro_tts.cli
assert not any(name in sys.modules for name in ('numpy', 'mlx_audio', 'mlx.core', 'huggingface_hub'))
print('Import gate passed without synthesis dependencies or network.')
'''
    subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT, check=True, timeout=20)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
