"""fire_rule.py 테스트 (합성 소리 사용 — 실제 녹음 성능을 뜻하지 않음).

실행: python -m unittest test_fire_rule -v
"""
import unittest

import numpy as np

from fire_rule import FireRuleDetector, analyze
from common import SR

rng = np.random.default_rng(0)


def tone(freq, dur, amp=0.3):
    t = np.arange(int(SR * dur)) / SR
    y = amp * np.sin(2 * np.pi * freq * t)
    fade = min(len(y) // 2, int(SR * 0.005))          # 딸깍 소리 방지
    if fade:
        ramp = np.linspace(0, 1, fade)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return y


def silence(dur):
    return np.zeros(int(SR * dur))


def t3(freq=3100, cycles=2, amp=0.3):
    one = [tone(freq, 0.5, amp), silence(0.5), tone(freq, 0.5, amp), silence(0.5),
           tone(freq, 0.5, amp), silence(1.5)]
    return np.concatenate(one * cycles)


def t4(freq=3100, amp=0.3):
    beeps = [part for _ in range(4) for part in (tone(freq, 0.1, amp), silence(0.1))]
    return np.concatenate(beeps + [silence(1.0)])


def white(dur, amp):
    return rng.normal(0, amp, int(SR * dur))


def babble(dur, amp=0.1):
    """TV·대화 비슷한 소리: 기본음이 바뀌는 배음 묶음."""
    out = []
    for _ in range(int(dur / 0.2)):
        f0 = rng.uniform(100, 250)
        t = np.arange(int(SR * 0.2)) / SR
        out.append(sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 21)))
    y = np.concatenate(out)
    return amp * y / np.max(np.abs(y))


def doorbell():
    """딩동: 낮은 음 두 개가 감쇠 (배음 포함)."""
    def ding(f, dur):
        t = np.arange(int(SR * dur)) / SR
        return np.exp(-4 * t) * sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 2, 3))
    return 0.3 * np.concatenate([ding(660, 0.6), ding(523, 0.9), silence(0.5)])


def mix(signal, noise):
    n = max(len(signal), len(noise))
    return np.pad(signal, (0, n - len(signal))) + np.pad(noise, (0, n - len(noise)))


class DetectsAlarm(unittest.TestCase):
    def test_t3_clean(self):
        r = analyze(t3())
        self.assertEqual((r["level"], r["pattern"]), ("CONFIRMED", "T3"))

    def test_t4_clean(self):
        r = analyze(t4())
        self.assertEqual((r["level"], r["pattern"]), ("CONFIRMED", "T4"))

    def test_t3_other_alarm_pitches(self):
        for f in (2900, 3300, 3500):
            with self.subTest(freq=f):
                self.assertEqual(analyze(t3(freq=f))["level"], "CONFIRMED")

    def test_t3_quiet_far_away(self):
        self.assertEqual(analyze(t3(amp=0.02))["level"], "CONFIRMED")

    def test_t3_with_white_noise(self):
        y = t3(amp=0.1)
        self.assertEqual(analyze(mix(y, white(len(y) / SR, 0.05)))["level"], "CONFIRMED")

    def test_t3_with_tv_babble(self):
        y = t3(amp=0.1)
        self.assertEqual(analyze(mix(y, babble(len(y) / SR, 0.2)))["level"], "CONFIRMED")

    def test_two_beeps_is_candidate(self):
        y = np.concatenate([tone(3100, 0.5), silence(0.5), tone(3100, 0.5), silence(0.5)])
        self.assertEqual(analyze(y)["level"], "CANDIDATE")


class IgnoresOtherSounds(unittest.TestCase):
    def test_silence(self):
        self.assertIsNone(analyze(silence(5))["level"])

    def test_white_noise(self):
        self.assertIsNone(analyze(white(5, 0.1))["level"])

    def test_babble(self):
        self.assertIsNone(analyze(babble(6, 0.3))["level"])

    def test_doorbell(self):
        self.assertIsNone(analyze(np.tile(doorbell(), 3))["level"])

    def test_same_rhythm_low_pitch(self):
        self.assertIsNone(analyze(t3(freq=1000))["level"])

    def test_single_beep(self):
        y = np.concatenate([silence(1), tone(3100, 0.5), silence(2)])
        self.assertIsNone(analyze(y)["level"])

    def test_continuous_tone_is_only_candidate(self):
        # 주전자 휘파람처럼 계속 울리는 고음은 확정하지 않는다
        r = analyze(np.concatenate([tone(3100, 3.0), silence(0.5)]))
        self.assertEqual((r["level"], r["pattern"]), ("CANDIDATE", "CONTINUOUS"))


class Streaming(unittest.TestCase):
    def test_alert_timing(self):
        """0.1초씩 넣을 때 소리 시작부터 몇 초 만에 알림이 나오는지."""
        y = mix(t3(amp=0.1), babble(8, 0.1))
        det, hop = FireRuleDetector(), int(SR * 0.1)
        first = {}
        for i in range(0, len(y), hop):
            level = det.feed(y[i:i + hop])["level"]
            first.setdefault(level, (i + hop) / SR)
        self.assertLessEqual(first["CANDIDATE"], 1.5)   # 두 번째 울림 직후
        self.assertLessEqual(first["CONFIRMED"], 2.5)   # 세 번째 울림 직후


if __name__ == "__main__":
    unittest.main()
