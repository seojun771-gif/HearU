"""분류 결과를 UART로 FPGA(EPS23)에 보낸다.

패킷 (5바이트 고정, FPGA에서 파싱하기 쉬운 형식)
    [0] 0xAA            시작 표시
    [1] class_id        0~5 (shared/labels.json 기준: 0 UNKNOWN, 1 FIRE, 2 BABY_CRY, 3 DOORBELL, 4 KNOCK, 5 CALL)
    [2] confidence      0~100
    [3] priority        1~6 (1이 가장 높음)
    [4] checksum        앞 4바이트의 XOR
UART 설정: 115200 bps, 8N1 (기본값, --baud로 변경)

실행 예
    python serial_bridge.py --list-ports               # 사용 가능한 포트 확인
    python serial_bridge.py --port COM3 --test 3       # 마이크 없이 DOORBELL(3) 패킷 1개 전송
    python serial_bridge.py --port COM3 --file x.wav   # 파일 분류 결과를 전송
    python serial_bridge.py --port COM3 --mic          # 마이크 실시간 분류 -> 전송
    python serial_bridge.py --dry-run --mic            # 포트 없이 패킷(hex)만 출력 (FPGA 없이 점검)

기본 동작: 신뢰도 80% 이상(CONFIRMED)만 보낸다. 60~80%(WEAK)도 보내려면 --weak.
"""
import argparse
import sys

from common import LABELS, UNKNOWN_ID, load_wav

START = 0xAA


def pack(class_id: int, confidence: float, priority: int) -> bytes:
    """confidence는 0~1 또는 0~100 모두 받는다(1 이하이면 0~1로 보고 100을 곱함)."""
    conf = round(confidence * 100) if confidence <= 1.0 else round(confidence)
    body = bytes([START, int(class_id), max(0, min(100, conf)), int(priority)])
    chk = 0
    for b in body:
        chk ^= b
    return body + bytes([chk])


def unpack(packet: bytes) -> dict:
    """검증용. 잘못된 패킷이면 ValueError."""
    if len(packet) != 5 or packet[0] != START:
        raise ValueError("길이 또는 시작 바이트가 맞지 않음")
    chk = 0
    for b in packet[:4]:
        chk ^= b
    if chk != packet[4]:
        raise ValueError("체크섬 불일치")
    return {"class_id": packet[1], "confidence": packet[2], "priority": packet[3]}


class Sender:
    def __init__(self, port, baud, dry_run, send_weak):
        self.dry_run, self.send_weak = dry_run, send_weak
        self.ser = None
        if not dry_run:
            import serial
            self.ser = serial.Serial(port, baud, timeout=1)

    def send_result(self, r: dict) -> None:
        if r["class_id"] == UNKNOWN_ID or (r["level"] == "WEAK" and not self.send_weak):
            return                      # 기타(UNKNOWN)는 알림 대상이 아님
        self.send(pack(r["class_id"], r["confidence"], r["priority"]))

    def send(self, packet: bytes) -> None:
        print("  TX", packet.hex(" ").upper(), "(dry-run)" if self.dry_run else "")
        if self.ser:
            self.ser.write(packet)
            self.ser.flush()

    def close(self):
        if self.ser:
            self.ser.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="예: COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--test", type=int, metavar="CLASS_ID", help="마이크 없이 해당 class_id 패킷 1개 전송")
    ap.add_argument("--file", help="wav 파일을 분류해서 전송")
    ap.add_argument("--mic", action="store_true", help="마이크 실시간 분류 후 전송")
    ap.add_argument("--weak", action="store_true", help="신뢰도 60~80%%(WEAK)도 전송")
    ap.add_argument("--dry-run", action="store_true", help="포트를 열지 않고 패킷만 출력")
    a = ap.parse_args()

    if a.list_ports:
        from serial.tools import list_ports
        ports = list(list_ports.comports())
        for p in ports:
            print(f"{p.device}  {p.description}")
        if not ports:
            print("연결된 시리얼 포트가 없습니다. 보드 USB 연결과 드라이버를 확인하세요.")
        return

    if not (a.test is not None or a.file or a.mic):
        ap.print_help()
        return
    if not a.dry_run and not a.port:
        sys.exit("--port COM3 처럼 포트를 지정하세요. (--list-ports로 확인, 포트 없이 점검은 --dry-run)")

    tx = Sender(a.port, a.baud, a.dry_run, a.weak)
    try:
        if a.test is not None:
            lab = LABELS[a.test]
            print(f"테스트 전송: {lab['name']} ({lab['ko']})")
            tx.send(pack(a.test, 0.95, lab["priority"]))
        elif a.file:
            from infer import classify
            r = classify(load_wav(a.file))
            print(r)
            tx.send_result(r)
        else:
            from infer import run_mic
            run_mic(on_result=tx.send_result)
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        tx.close()


if __name__ == "__main__":
    main()
