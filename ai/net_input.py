"""Wi-Fi(UDP)로 장치가 보내는 PCM을 받아 분류하고, 결과를 장치에 돌려보낸다.

장치 -> PC  UDP 패킷: [seq 2바이트, 빅엔디언, 0~65535 순환] + [PCM 샘플들: int16, 리틀엔디언, mono]
            (기본: 16kHz, 패킷당 320샘플 = 20ms = 644바이트. 샘플 수는 가변이어도 된다)
PC -> 장치  UDP 패킷: serial_bridge.py와 같은 5바이트 [0xAA, class_id, 신뢰도, 우선순위, 체크섬]
            (PCM을 보낸 장치의 주소/포트로 그대로 회신)

실행:  python net_input.py                      # 포트 5005에서 대기
       python net_input.py --port 5005 --weak   # 60~80%(WEAK)도 회신
       python net_input.py --sr 8000            # 장치 샘플링 레이트가 16kHz가 아닐 때 (16kHz로 변환)

장치 없이 시험:  (터미널 1) python net_input.py     (터미널 2) python fake_device.py data/doorbell/doorbell_20261117_01.wav

동작: infer.py --mic과 비슷하다. 최근 1.5초를 0.5초마다 분류하고, 무음이면 건너뛰고,
      UNKNOWN이거나 같은 클래스가 쿨다운(5초) 안에 또 나오면 회신하지 않는다.
      차이점: (1) 무음 판단을 20ms 구간 최대 음량으로 한다(--gate), (2) 같은 결과가 연속 2번
      나와야 회신한다(--confirm). 소리가 막 시작될 때의 순간적 오분류를 거르기 위해서다.
방화벽: Windows가 처음 실행할 때 네트워크 허용을 묻는다. 허용해야 장치에서 도달한다.
"""
import argparse
import socket
import time

import numpy as np

from common import SR, UNKNOWN_ID, _cfg
from infer import classify, to_uart
from serial_bridge import pack

MAX_GAP_PACKETS = 500          # 이보다 많이 건너뛰면 끊긴 것으로 보고 버퍼를 새로 시작


def local_ips() -> list:
    """이 PC의 IP 후보 (장치에 알려 줄 주소)."""
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


def resample(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return y
    import librosa
    return librosa.resample(y, orig_sr=sr, target_sr=SR)


def peak_frame_rms(y: np.ndarray, frame: int = SR // 50) -> float:
    """20ms 구간별 RMS 중 최댓값 (평균 RMS는 짧은 노크를 무음으로 만든다)."""
    n = len(y) // frame
    if n == 0:
        return 0.0
    return float(np.sqrt(np.mean(y[:n * frame].reshape(n, frame) ** 2, axis=1)).max())


def run(port: int, dev_sr: int, send_weak: bool, rms_gate: float, confirm_n: int = 2,
        host: str = "0.0.0.0", max_seconds=None, verbose: bool = True):
    cooldown = _cfg["cooldown_sec"]
    t = time.perf_counter()
    classify(np.random.default_rng(0).normal(0, 0.1, SR).astype(np.float32))   # 모델 로드·librosa 초기화(첫 호출이 느림)
    print(f"준비 완료 (워밍업 {time.perf_counter() - t:.1f}초)")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.5)
    print(f"UDP {port}번 포트에서 대기 중 (장치 샘플링 레이트 {dev_sr}Hz). 장치에 알려 줄 PC 주소 후보: {local_ips()}")

    win = int(dev_sr * 1.5)               # 분류에 쓰는 1.5초 (장치 샘플 기준)
    hop = dev_sr // 2                     # 0.5초마다 분류
    buf = np.zeros(win, dtype=np.float32)
    expected, since, lost, got = None, 0, 0, 0
    streak = (None, 0)
    sender, late = None, 0
    last = {}
    stat_t, stat_n, stat_peak = time.time(), 0, 0.0      # 도착 확인용 통계
    t_start = time.time()

    try:
        while True:
            if max_seconds and time.time() - t_start > max_seconds:
                break
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            if len(data) < 4:
                continue
            seq = (data[0] << 8) | data[1]
            pcm = np.frombuffer(data[2:len(data) - (len(data) - 2) % 2], dtype="<i2").astype(np.float32) / 32768.0
            got += 1
            if verbose:
                if addr != sender:
                    print(f"* 장치 연결됨: {addr[0]}:{addr[1]} (첫 패킷 seq={seq}, 샘플 {len(pcm)}개 = {len(pcm) / dev_sr * 1000:.0f}ms)")
                    stat_t, stat_n, stat_peak = time.time(), 0, 0.0
                stat_n += 1
                stat_peak = max(stat_peak, float(np.abs(pcm).max()) if len(pcm) else 0.0)
                if time.time() - stat_t >= 5:
                    print(f"  [수신 중] 최근 5초 패킷 {stat_n}개 (정상이면 약 {int(5 * dev_sr / max(len(pcm), 1))}개), 최대 음량 {stat_peak:.3f}, 누적 유실 {lost}")
                    stat_t, stat_n, stat_peak = time.time(), 0, 0.0

            if addr != sender:                         # 새 장치(또는 재접속): 번호·버퍼 새로 시작
                sender, expected, late, buf, since, streak = addr, None, 0, np.zeros(win, dtype=np.float32), 0, (None, 0)
            if expected is not None:
                gap = (seq - expected) & 0xFFFF
                if gap >= 32768:                       # 이미 지난 패킷(순서 뒤바뀜)
                    late += 1
                    if late < 5:                       # 가끔이면 버리고, 계속 이러면 장치가 재시작한 것으로 보고 새로 시작
                        continue
                    expected, late, buf, since, streak = None, 0, np.zeros(win, dtype=np.float32), 0, (None, 0)
                    gap = 0
                else:
                    late = 0
            if expected is not None:
                if 0 < gap < MAX_GAP_PACKETS:          # 빠진 구간은 무음으로 채움
                    fill = np.zeros(gap * len(pcm), dtype=np.float32)
                    buf = np.concatenate([buf, fill])[-win:]
                    since += len(fill)
                    lost += gap
                elif gap >= MAX_GAP_PACKETS:
                    buf[:] = 0
            expected = (seq + 1) & 0xFFFF

            buf = np.concatenate([buf, pcm])[-win:]
            since += len(pcm)
            if since < hop:
                continue
            since = 0

            y = resample(buf, dev_sr)
            if peak_frame_rms(y) < rms_gate:           # 무음 (노크 같은 짧은 소리도 잡도록 20ms 구간 최댓값 기준)
                streak = (None, 0)
                continue
            t0 = time.perf_counter()
            r = classify(y)
            ms = (time.perf_counter() - t0) * 1000
            if r["class_id"] == UNKNOWN_ID or (r["level"] == "WEAK" and not send_weak):
                streak = (None, 0)
                continue                               # 기타(UNKNOWN)는 LED 알림 대상이 아님
            # 소리가 막 시작될 때의 순간적 오분류를 거르기 위해 같은 결과가 연속 confirm_n번 나와야 회신
            streak = (r["class_id"], streak[1] + 1) if streak[0] == r["class_id"] else (r["class_id"], 1)
            if streak[1] < confirm_n:
                continue
            if time.time() - last.get(r["label"], 0) < cooldown:
                continue
            last[r["label"]] = time.time()
            packet = pack(r["class_id"], r["confidence"], r["priority"])
            sock.sendto(packet, addr)
            print(f"[{r['level']}] {r['ko']} {r['confidence']:.2f} ({ms:.0f}ms) {to_uart(r).strip()} "
                  f"-> 회신 {packet.hex(' ').upper()} to {addr[0]}:{addr[1]} (받은 패킷 {got}, 유실 {lost})")
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        sock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--sr", type=int, default=SR, help="장치가 보내는 샘플링 레이트 (기본 16000)")
    ap.add_argument("--weak", action="store_true", help="신뢰도 60~80%%(WEAK)도 회신")
    ap.add_argument("--gate", type=float, default=0.005, help="20ms 구간 최대 음량이 이보다 작으면 무음 처리 (기본 0.005 ≈ -46dBFS)")
    ap.add_argument("--confirm", type=int, default=2, help="같은 결과가 연속 몇 번 나와야 회신할지 (기본 2, 1이면 즉시)")
    ap.add_argument("--quiet", action="store_true", help="도착 확인 메시지(연결·5초 통계)를 끈다")
    ap.add_argument("--seconds", type=float, help="이 시간(초) 후 종료 (시험용)")
    a = ap.parse_args()
    run(a.port, a.sr, a.weak, a.gate, a.confirm, max_seconds=a.seconds, verbose=not a.quiet)


if __name__ == "__main__":
    main()
