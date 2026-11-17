"""가짜 장치: wav 파일을 장치처럼 UDP로 실시간 전송하고, PC가 돌려주는 5바이트 결과를 출력한다.
(EPS23 없이 net_input.py를 시험하기 위한 도구. 팀원이 장치 코드를 만들 때 패킷 형식 참고용으로도 쓴다.)

실행:  python fake_device.py data/doorbell/doorbell_20261117_01.wav
       python fake_device.py x.wav --host 192.168.0.10 --port 5005 --lead 2 --tail 4

전송 형식: [seq 2바이트 빅엔디언] + [int16 리틀엔디언 샘플 320개(20ms)], 실제 시간 속도(20ms마다)로 보낸다.
파일 앞에 --lead초, 뒤에 --tail초 무음을 붙여 보낸다.
"""
import argparse
import socket
import time

import numpy as np

from common import LABELS, SR, load_wav

CHUNK = 320   # 20ms @ 16kHz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--lead", type=float, default=1.0, help="앞 무음(초)")
    ap.add_argument("--tail", type=float, default=3.0, help="뒤 무음(초). 결과를 기다리는 시간이기도 함")
    ap.add_argument("--drop", type=float, default=0.0, help="패킷 유실률 시험 (0~1)")
    a = ap.parse_args()

    y = np.concatenate([np.zeros(int(a.lead * SR)), load_wav(a.wav), np.zeros(int(a.tail * SR))]).astype(np.float32)
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    pcm = np.pad(pcm, (0, (-len(pcm)) % CHUNK))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0)
    rng = np.random.default_rng(0)
    print(f"{a.host}:{a.port}로 {len(pcm) / SR:.1f}초 전송 시작")
    t0 = time.perf_counter()
    for i in range(len(pcm) // CHUNK):
        due = t0 + i * CHUNK / SR
        while time.perf_counter() < due:
            try:                                           # 기다리는 동안 회신 확인
                data, _ = sock.recvfrom(64)
                show(data, time.perf_counter() - t0)
            except (BlockingIOError, socket.timeout):
                time.sleep(0.001)
        if rng.random() < a.drop:
            continue
        sock.sendto(bytes([(i >> 8) & 0xFF, i & 0xFF]) + pcm[i * CHUNK:(i + 1) * CHUNK].tobytes(), (a.host, a.port))
    time.sleep(0.5)
    try:
        while True:
            data, _ = sock.recvfrom(64)
            show(data, time.perf_counter() - t0)
    except (BlockingIOError, socket.timeout):
        pass
    print("전송 끝")


def show(data: bytes, t: float):
    ok = len(data) == 5 and data[0] == 0xAA and (data[0] ^ data[1] ^ data[2] ^ data[3]) == data[4]
    if ok:
        lab = LABELS.get(data[1], {"name": "?", "ko": "?"})
        print(f"  [{t:5.2f}s] 수신 {data.hex(' ').upper()}  -> {lab['name']}({lab['ko']}) 신뢰도 {data[2]}% 우선순위 {data[3]}")
    else:
        print(f"  [{t:5.2f}s] 잘못된 회신: {data.hex(' ')}")


if __name__ == "__main__":
    main()
