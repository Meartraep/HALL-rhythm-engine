import os
import shutil
import subprocess
from bisect import bisect_right

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection
import numpy as np
from pydub import AudioSegment
import soundfile as sf

# ===================== User Configuration =====================
FPS = int(input('Enter render FPS (e.g. 60, 120, 240): '))
OUTPUT_SIDE_PX = int(input('Enter output square side length in pixels (e.g. 1200): '))
TOTAL_DURATION = 370.0
AUDIO_SAMPLE_RATE = 44100
CLICK_BASE_GAIN_DB = -6.0
CLICK_RELATIVE_MARGIN_DB = 10.0
CLICK_TARGET_MIN_PEAK_DBFS = -8.0
CLICK_TARGET_MAX_PEAK_DBFS = -1.5
CLICK_GAIN_SMOOTH_TAU = 0.35
CLICK_ENVELOPE_WINDOW_MS = 320.0
CLICK_ENVELOPE_HOP_MS = 40.0
CLICK_SONG_ANALYSIS_FLOOR_DB = -80.0
RENDER_DPI = 120
FIGSIZE_INCH = OUTPUT_SIDE_PX / RENDER_DPI

# ===================== Time Signature Data =====================
RAW_TIME_SIGNATURES = [
    (0.0,      (4, 4),   17.884),
    (17.884,   (7, 8),   18.759),
    (18.759,   'loop',   47.759),
    (47.759,   (5, 4),  118.592),
    (118.592,  (9, 8),  124.592),
    (124.592,  (7, 8),  127.894),
    (127.894,  (11, 8), 129.488),
    (129.488,  (13, 8), 131.270),
    (131.270,  (7, 8),  195.020),
    (195.020,  (6, 4),  307.235),
    (307.235,  (8, 4),  317.735),
    (317.735,  (9, 4),  319.422),
    (319.422,  (8, 4),  370.0),
]
LOOP_SIGNATURES = [(4, 4), (3, 4), (4, 4), (7, 8), (4, 4), (5, 8), (4, 4), (4, 4)] * 4
LOOP_START = 18.759
LOOP_END = 47.759
TOTAL_LOOP_MEASURES = 32

# ===================== BPM Data (Precise Tempo Mapping) =====================
BPM_DATA = [
    (0.0,      210,             9.143),
    (9.143,    215,            13.608),
    (13.608,   220,            15.790),
    (15.790,   225,            16.856),
    (16.856,   'beat_split',   55.259),
    (55.259,   180,           124.592),
    (124.592,  'increase_8th', 129.488),
    (129.488,  224,           195.020),
    (195.020,  246,           288.679),
    (288.679,  'decrease_24th', 301.780),
    (301.780,  66,            307.235),
    (307.235,  320,           370.0),
]

# ===================== Precomputed Step Tempo Maps =====================
def build_step_tempo_map(start_time, end_time, bpm_start, bpm_end, subdivision):
    times = [start_time]
    bpms = [bpm_start]
    t = start_time
    bpm = bpm_start
    while t < end_time and bpm != bpm_end:
        step_duration = 60.0 / (bpm * subdivision)
        t += step_duration
        bpm += 1 if bpm_end > bpm_start else -1
        if t <= end_time + 1e-9:
            times.append(t)
            bpms.append(bpm)
    return np.array(times, dtype=float), np.array(bpms, dtype=int)

INCREASE_STEP_TIMES, INCREASE_STEP_BPMS = build_step_tempo_map(124.592, 129.488, 180, 224, 2.0)
DECREASE_SUBSTEP_TIMES, DECREASE_SUBSTEP_BPMS = build_step_tempo_map(288.679, 301.780, 246, 66, 6.0)

# ===================== Real-Time BPM Getter =====================
def get_current_bpm(t):
    for start, bpm_type, end in BPM_DATA:
        if start <= t < end:
            if isinstance(bpm_type, int):
                return bpm_type
            if bpm_type == 'beat_split':
                local = t - start
                beat_len = 60.0 / 230.0
                beat_idx = int(local / beat_len)
                phase = beat_idx % 4
                if phase == 0:
                    return 230
                elif phase == 2:
                    return 235
                return 240
            if bpm_type == 'increase_8th':
                idx = bisect_right(INCREASE_STEP_TIMES, t) - 1
                idx = max(0, min(idx, len(INCREASE_STEP_BPMS) - 1))
                return int(INCREASE_STEP_BPMS[idx])
            if bpm_type == 'decrease_24th':
                idx = bisect_right(DECREASE_SUBSTEP_TIMES, t) - 1
                idx = max(0, min(idx, len(DECREASE_SUBSTEP_BPMS) - 1))
                return int(DECREASE_SUBSTEP_BPMS[idx])
    return 210

# ===================== Expanded Time Signature Segments =====================
def expand_time_signatures():
    segs = []
    for start, sig, end in RAW_TIME_SIGNATURES:
        if sig == 'loop':
            duration = end - start
            per_sig = duration / len(LOOP_SIGNATURES)
            for i, (num, den) in enumerate(LOOP_SIGNATURES):
                segs.append((start + i * per_sig, start + (i + 1) * per_sig, num, den))
        else:
            segs.append((start, end, sig[0], sig[1]))
    return segs

TS_SEGS = expand_time_signatures()

def get_time_sig_at(t):
    for start, end, num, den in TS_SEGS:
        if start <= t < end:
            return start, end, num, den
    return TS_SEGS[-1][0], TS_SEGS[-1][1], TS_SEGS[-1][2], TS_SEGS[-1][3]

# ===================== Global Beat Integration =====================
def build_global_beat_map():
    events = []
    for bpm_start, bpm_type, bpm_end in BPM_DATA:
        if isinstance(bpm_type, int):
            events.append((bpm_start, 'const', bpm_type))
            events.append((bpm_end, 'end', None))
        elif bpm_type == 'beat_split':
            t = bpm_start
            beat_idx = 0
            while t < bpm_end:
                phase = beat_idx % 4
                current_bpm = 230 if phase == 0 else (235 if phase == 2 else 240)
                events.append((t, 'const', current_bpm))
                beat_dur = 60.0 / current_bpm
                t += beat_dur
                beat_idx += 1
            events.append((bpm_end, 'end', None))
        elif bpm_type == 'increase_8th':
            t = bpm_start
            current_bpm = 180
            while t < bpm_end and current_bpm <= 224:
                events.append((t, 'const', current_bpm))
                note_dur = 60.0 / current_bpm
                t += note_dur
                current_bpm += 1
            events.append((bpm_end, 'end', None))
        elif bpm_type == 'decrease_24th':
            t = bpm_start
            current_bpm = 246
            while t < bpm_end and current_bpm >= 66:
                events.append((t, 'const', current_bpm))
                note_dur = (60.0 / current_bpm) / 6.0
                t += note_dur
                current_bpm -= 1
            events.append((bpm_end, 'end', None))

    ts_times = [s for s, e, n, d in TS_SEGS] + [e for s, e, n, d in TS_SEGS]
    all_times = sorted(list(set([t for t, _, _ in events] + ts_times + [0.0, TOTAL_DURATION])))

    beat_map = [(0.0, 0.0)]
    for i in range(1, len(all_times)):
        t_prev, q_prev = beat_map[-1]
        t_curr = all_times[i]
        if t_curr > TOTAL_DURATION:
            break

        current_bpm = 210
        for j in range(len(events) - 1):
            if events[j][0] <= t_prev < events[j + 1][0]:
                if events[j][1] == 'const':
                    current_bpm = events[j][2]
                break

        q_per_sec = current_bpm / 60.0
        dt = t_curr - t_prev
        dq = q_per_sec * dt
        q_curr = q_prev + dq
        beat_map.append((t_curr, q_curr))
    return beat_map

GLOBAL_BEAT_MAP = build_global_beat_map()
GBM_TIMES = [t for t, q in GLOBAL_BEAT_MAP]
GBM_QUARTERS = [q for t, q in GLOBAL_BEAT_MAP]


def get_global_quarter_beat(t):
    idx = bisect_right(GBM_TIMES, t) - 1
    idx = max(0, min(idx, len(GLOBAL_BEAT_MAP) - 2))
    t0, q0 = GLOBAL_BEAT_MAP[idx]
    t1, q1 = GLOBAL_BEAT_MAP[idx + 1]
    if t1 == t0:
        return q0
    return q0 + (q1 - q0) * (t - t0) / (t1 - t0)


def time_from_global_quarter(q_target):
    if q_target <= GBM_QUARTERS[0]:
        return GBM_TIMES[0]
    if q_target >= GBM_QUARTERS[-1]:
        return GBM_TIMES[-1]
    idx = bisect_right(GBM_QUARTERS, q_target) - 1
    idx = max(0, min(idx, len(GLOBAL_BEAT_MAP) - 2))
    t0, q0 = GLOBAL_BEAT_MAP[idx]
    t1, q1 = GLOBAL_BEAT_MAP[idx + 1]
    if q1 == q0:
        return t0
    return t0 + (t1 - t0) * (q_target - q0) / (q1 - q0)

# ===================== Beat Click Events =====================
def build_click_events():
    events = []

    for measure_num, start_time, end_time, num, den in MEASURE_MAP:
        if end_time <= start_time:
            continue

        beat_duration = (end_time - start_time) / num
        for k in range(num):
            t_event = start_time + k * beat_duration
            events.append((t_event, k == 0))

    events.sort(key=lambda x: x[0])
    dedup = []
    last_t = None
    for t, is_downbeat in events:
        if last_t is None or abs(t - last_t) > 1e-5:
            dedup.append((t, is_downbeat))
            last_t = t
        else:
            if is_downbeat:
                dedup[-1] = (t, True)
    return dedup


# ===================== Click Track Rendering =====================
def _audiosegment_to_float32(seg):
    seg = seg.set_frame_rate(AUDIO_SAMPLE_RATE).set_channels(2)
    samples = np.array(seg.get_array_of_samples())
    if seg.channels == 2:
        samples = samples.reshape((-1, 2))
    else:
        samples = np.repeat(samples[:, None], 2, axis=1)
    max_val = float(1 << (8 * seg.sample_width - 1))
    return samples.astype(np.float32) / max_val, seg.frame_rate


def _audiosegment_to_float32_mono(seg):
    seg = seg.set_frame_rate(AUDIO_SAMPLE_RATE).set_channels(1)
    samples = np.array(seg.get_array_of_samples())
    max_val = float(1 << (8 * seg.sample_width - 1))
    return samples.astype(np.float32) / max_val, seg.frame_rate


def _build_song_loudness_envelope(audio_path):
    song_seg = AudioSegment.from_file(audio_path)
    song_arr, sr = _audiosegment_to_float32_mono(song_seg)
    if sr != AUDIO_SAMPLE_RATE:
        raise RuntimeError('Unexpected audio sample rate conversion failure.')

    if song_arr.size == 0:
        times = np.array([0.0], dtype=float)
        env_db = np.array([CLICK_SONG_ANALYSIS_FLOOR_DB], dtype=float)
        return times, env_db

    window_samples = max(1, int(round(CLICK_ENVELOPE_WINDOW_MS * sr / 1000.0)))
    hop_samples = max(1, int(round(CLICK_ENVELOPE_HOP_MS * sr / 1000.0)))
    half_window = window_samples // 2

    sq = np.square(song_arr.astype(np.float64))
    csum = np.concatenate(([0.0], np.cumsum(sq, dtype=np.float64)))

    centers = np.arange(0, len(song_arr), hop_samples, dtype=np.int64)
    env_raw = np.empty(len(centers), dtype=np.float32)

    for i, center in enumerate(centers):
        start = max(0, int(center) - half_window)
        end = min(len(song_arr), int(center) + half_window)
        n = max(end - start, 1)
        mean_sq = (csum[end] - csum[start]) / n
        db = 10.0 * np.log10(max(mean_sq, 1e-12))
        env_raw[i] = max(db, CLICK_SONG_ANALYSIS_FLOOR_DB)

    smoothed = np.empty_like(env_raw, dtype=np.float32)
    smoothed[0] = env_raw[0]
    step_dt = hop_samples / sr
    attack_tau = 0.18
    release_tau = 0.70

    for i in range(1, len(env_raw)):
        prev = float(smoothed[i - 1])
        current = float(env_raw[i])
        tau = attack_tau if current > prev else release_tau
        alpha = 1.0 - np.exp(-step_dt / tau)
        smoothed[i] = prev + alpha * (current - prev)

    times = centers.astype(np.float32) / float(sr)
    return times.astype(np.float64), smoothed.astype(np.float64)


def _interp_envelope_value(times, values, t):
    if t <= times[0]:
        return float(values[0])
    if t >= times[-1]:
        return float(values[-1])
    return float(np.interp(t, times, values))


def build_click_track(script_dir, audio_path, output_wav):
    first_path = os.path.join(script_dir, 'First.mp3')
    clap_path = os.path.join(script_dir, 'clap.mp3')

    if not os.path.isfile(first_path):
        raise FileNotFoundError(f'Missing click sample: {first_path}')
    if not os.path.isfile(clap_path):
        raise FileNotFoundError(f'Missing click sample: {clap_path}')
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f'Missing audio file: {audio_path}')

    env_times, env_db = _build_song_loudness_envelope(audio_path)

    first_seg = AudioSegment.from_file(first_path)
    clap_seg = AudioSegment.from_file(clap_path)

    first_arr_base, sr1 = _audiosegment_to_float32(first_seg.apply_gain(CLICK_BASE_GAIN_DB))
    clap_arr_base, sr2 = _audiosegment_to_float32(clap_seg.apply_gain(CLICK_BASE_GAIN_DB))
    if sr1 != AUDIO_SAMPLE_RATE or sr2 != AUDIO_SAMPLE_RATE:
        raise RuntimeError('Unexpected audio sample rate conversion failure.')

    first_peak = float(np.max(np.abs(first_arr_base)))
    clap_peak = float(np.max(np.abs(clap_arr_base)))
    first_peak_dbfs = 20.0 * np.log10(max(first_peak, 1e-12))
    clap_peak_dbfs = 20.0 * np.log10(max(clap_peak, 1e-12))

    max_len = max(len(first_arr_base), len(clap_arr_base))
    total_samples = int((TOTAL_DURATION + 2.0) * AUDIO_SAMPLE_RATE) + max_len + 1
    mix = np.zeros((total_samples, 2), dtype=np.float32)

    prev_gain_db = None
    prev_t = None

    for t, is_downbeat in CLICK_EVENTS:
        song_db = _interp_envelope_value(env_times, env_db, t)

        target_peak_dbfs = song_db + CLICK_RELATIVE_MARGIN_DB
        target_peak_dbfs = float(np.clip(
            target_peak_dbfs,
            CLICK_TARGET_MIN_PEAK_DBFS,
            CLICK_TARGET_MAX_PEAK_DBFS,
        ))

        src = first_arr_base if is_downbeat else clap_arr_base
        src_peak_dbfs = first_peak_dbfs if is_downbeat else clap_peak_dbfs
        desired_gain_db = target_peak_dbfs - src_peak_dbfs

        if prev_gain_db is None:
            smoothed_gain_db = desired_gain_db
        else:
            dt = max(t - prev_t, 0.0)
            alpha = 1.0 - np.exp(-dt / CLICK_GAIN_SMOOTH_TAU)
            smoothed_gain_db = prev_gain_db + alpha * (desired_gain_db - prev_gain_db)

        prev_gain_db = smoothed_gain_db
        prev_t = t

        gain = 10.0 ** (smoothed_gain_db / 20.0)
        arr = src * gain

        start = int(round(t * AUDIO_SAMPLE_RATE))
        end = min(start + len(arr), total_samples)
        if start < 0 or start >= total_samples:
            continue
        mix[start:end] += arr[: end - start]

    np.clip(mix, -1.0, 1.0, out=mix)
    sf.write(output_wav, mix, AUDIO_SAMPLE_RATE, subtype='PCM_16')

# ===================== Measure Schedule (0-based) =====================
# These anchors are taken from the provided timing data and define the canonical
# 0-based measure timeline used by both the on-screen numbering and click track.
MEASURE_ANCHORS = [
    (0.0, 0),
    (17.884, 16),
    (18.759, 17),
    (47.759, 49),
    (118.592, 93),
    (124.592, 97),
    (127.894, 100),
    (129.488, 101),
    (131.270, 102),
    (195.020, 170),
    (307.235, 240),
    (317.735, 247),
    (319.422, 248),
]

MEASURE_SEGMENTS = [
    (MEASURE_ANCHORS[i][0], MEASURE_ANCHORS[i][1], MEASURE_ANCHORS[i + 1][0], MEASURE_ANCHORS[i + 1][1])
    for i in range(len(MEASURE_ANCHORS) - 1)
]

LAST_MEASURE_START_TIME = 319.422
LAST_MEASURE_START_NUMBER = 248
LAST_MEASURE_BPM = 320.0
LAST_MEASURE_NUM = 8
LAST_MEASURE_DEN = 4


def build_measure_map():
    measures = []

    # Anchor-to-anchor ranges are interpolated in measure space so the click track
    # and the displayed measure number use the same partition.
    for start_time, start_measure, end_time, end_measure in MEASURE_SEGMENTS:
        measure_count = int(end_measure - start_measure)
        if measure_count <= 0:
            continue

        interval = end_time - start_time
        for i in range(measure_count):
            m_num = start_measure + i
            frac0 = i / measure_count
            frac1 = (i + 1) / measure_count
            m_start = start_time + interval * frac0
            m_end = start_time + interval * frac1
            _, _, num, den = get_time_sig_at(m_start + 1e-9)
            measures.append((m_num, m_start, m_end, num, den))

    # The tail section after the last anchor is an explicit 8/4 region at 320 BPM.
    # Generate it directly so the bar numbering stays 0-based and contiguous.
    t = LAST_MEASURE_START_TIME
    m_num = LAST_MEASURE_START_NUMBER
    measure_duration = (60.0 / LAST_MEASURE_BPM) * LAST_MEASURE_NUM
    while t < TOTAL_DURATION - 1e-9:
        m_end = min(t + measure_duration, TOTAL_DURATION)
        measures.append((m_num, t, m_end, LAST_MEASURE_NUM, LAST_MEASURE_DEN))
        t = m_end
        m_num += 1

    return measures


MEASURE_MAP = build_measure_map()
MEASURE_START_TIMES = [m[1] for m in MEASURE_MAP]
CLICK_EVENTS = build_click_events()


def get_song_measure_number(t):
    if t < 0:
        return 0
    idx = bisect_right(MEASURE_START_TIMES, t) - 1
    idx = max(0, min(idx, len(MEASURE_MAP) - 1))
    return MEASURE_MAP[idx][0]


def get_measure_entry_at(t):
    if t < 0:
        return MEASURE_MAP[0], 0
    idx = bisect_right(MEASURE_START_TIMES, t) - 1
    idx = max(0, min(idx, len(MEASURE_MAP) - 1))
    return MEASURE_MAP[idx], idx

# ===================== Loop Measure Counter =====================
def get_loop_measure_index(t):
    if LOOP_START <= t <= LOOP_END:
        progress = (t - LOOP_START) / (LOOP_END - LOOP_START)
        current = int(np.floor(progress * TOTAL_LOOP_MEASURES)) + 1
        return min(current, TOTAL_LOOP_MEASURES)
    return 0

# ===================== Visualization =====================
def get_regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n + 1)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return x, y


def build_polygon_tick_segments(n, radius=1.0):
    x, y = get_regular_polygon(n, radius=radius)
    pts = np.column_stack([x[:-1], y[:-1]])
    segments = []
    tick_len = 0.07 * radius
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        edge = p1 - p0
        edge_norm = np.linalg.norm(edge)
        if edge_norm < 1e-12:
            continue
        tangent = edge / edge_norm
        outward = np.array([tangent[1], -tangent[0]])
        mid = (p0 + p1) / 2.0
        if np.dot(outward, mid) < 0:
            outward = -outward

        for frac, scale in ((1.0 / 3.0, 0.85), (0.5, 1.0), (2.0 / 3.0, 0.85)):
            p = p0 + frac * edge
            start = p - outward * tick_len * scale * 0.5
            end = p + outward * tick_len * scale * 0.5
            segments.append([start, end])
    return segments


# ===================== Figure Setup =====================
plt.rcParams['figure.figsize'] = [FIGSIZE_INCH, FIGSIZE_INCH]
fig, ax = plt.subplots(figsize=(FIGSIZE_INCH, FIGSIZE_INCH), dpi=RENDER_DPI)
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.axis('off')
ax.set_aspect('equal')

tick_collection = LineCollection([], colors='white', linewidths=1.0, alpha=0.35, zorder=4)
ax.add_collection(tick_collection)

polygon_line, = ax.plot([], [], 'w-', linewidth=3, alpha=0.3, zorder=5)
highlight_line, = ax.plot([], [], color='cyan', linewidth=6, zorder=6)
time_text = ax.text(0, 1.25, '', ha='center', fontsize=12, color='white')
sig_text = ax.text(0, -1.25, '', ha='center', fontsize=14, color='cyan')
bpm_text = ax.text(-1.3, 1.25, '', ha='left', fontsize=10, color='white')
measure_text = ax.text(1.3, 1.25, '', ha='right', fontsize=10, color='white')
loop_text = ax.text(0, 0, '', ha='center', fontsize=12, color='orange')
beat_dot, = ax.plot([], [], marker='o', markersize=11, color='yellow', markeredgecolor='white', markeredgewidth=1.2, linestyle='None', zorder=7)
phase_text = ax.text(0, 1.08, '', ha='center', fontsize=11, color='yellow')
beat_labels = []
TICK_SEGMENTS_CACHE = {}

# ===================== Animation Update =====================
def update(frame):
    global beat_labels
    t = frame / FPS

    (measure_num, measure_start, measure_end, num, den), measure_idx = get_measure_entry_at(t)
    if measure_end <= measure_start:
        measure_end = measure_start + 1e-9

    beat_span = measure_end - measure_start
    beat_pos = ((t - measure_start) / beat_span) * num if beat_span > 0 else 0.0
    beat_pos = max(0.0, min(beat_pos, max(num - 1e-9, 0.0)))
    beat_idx = int(np.floor(beat_pos)) % num
    beat_phase = beat_pos - np.floor(beat_pos)

    current_bpm = get_current_bpm(t)
    loop_measure = get_loop_measure_index(t)
    measure_number = measure_num

    x, y = get_regular_polygon(num)
    polygon_line.set_data(x, y)
    highlight_line.set_data([x[beat_idx], x[beat_idx + 1]], [y[beat_idx], y[beat_idx + 1]])

    if num not in TICK_SEGMENTS_CACHE:
        TICK_SEGMENTS_CACHE[num] = build_polygon_tick_segments(num)
    tick_collection.set_segments(TICK_SEGMENTS_CACHE[num])

    dot_x = x[beat_idx] + beat_phase * (x[beat_idx + 1] - x[beat_idx])
    dot_y = y[beat_idx] + beat_phase * (y[beat_idx + 1] - y[beat_idx])
    beat_dot.set_data([dot_x], [dot_y])

    for txt in beat_labels:
        txt.remove()
    beat_labels = []
    for i in range(num):
        txt = ax.text(
            x[i] * 1.15,
            y[i] * 1.15,
            f'{i + 1}',
            ha='center',
            va='center',
            fontsize=11,
            color='orange',
            weight='bold',
        )
        beat_labels.append(txt)

    time_text.set_text(f'Time: {t:.2f}s / {TOTAL_DURATION:.0f}s')
    sig_text.set_text(f'Signature: {num}/{den} | Beat: {beat_idx + 1}/{num}')
    bpm_text.set_text(f'BPM: {current_bpm}')
    measure_text.set_text(f'Measure: {measure_number:03d}')
    loop_text.set_text(f'Loop: {loop_measure}/{TOTAL_LOOP_MEASURES}' if loop_measure > 0 else '')
    phase_text.set_text(f'Beat position: {beat_phase * 100:.1f}% of current beat')

    return (
        polygon_line,
        highlight_line,
        beat_dot,
        time_text,
        sig_text,
        phase_text,
        bpm_text,
        measure_text,
        loop_text,
        *beat_labels,
    )

# ===================== Rendering =====================
def render_with_intel_qsv(ani, temp_video):
    print(f'Rendering at {FPS} FPS (Intel QSV)...')
    writer = FFMpegWriter(fps=FPS, codec='h264_qsv', bitrate=10000, extra_args=['-pix_fmt', 'nv12'])
    ani.save(temp_video, writer=writer, dpi=RENDER_DPI)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file = os.path.join(script_dir, 'HALL.flac')
    temp_video = os.path.join(script_dir, 'temp_video_no_audio.mp4')
    temp_click_wav = os.path.join(script_dir, 'temp_click_track.wav')
    temp_mix_audio = os.path.join(script_dir, 'temp_mixed_audio.m4a')
    final_output = os.path.join(script_dir, 'HALL_Rhythm_FINAL.mp4')

    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found in PATH.')
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f'Audio file missing: {audio_file}')

    print('Building click track...')
    build_click_track(script_dir, audio_file, temp_click_wav)

    print('Preparing animation...')
    ani = FuncAnimation(
        fig,
        update,
        frames=int(TOTAL_DURATION * FPS),
        interval=1000 / FPS,
        blit=False,
        cache_frame_data=False,
    )

    try:
        render_with_intel_qsv(ani, temp_video)
    except Exception as e:
        print(f'Intel QSV failed: {e}')
        print('Falling back to libx264...')
        writer = FFMpegWriter(fps=FPS, codec='libx264', bitrate=10000, extra_args=['-pix_fmt', 'yuv420p'])
        ani.save(temp_video, writer=writer, dpi=RENDER_DPI)

    plt.close(fig)
    print('Mixing audio with click track...')

    mix_cmd = [
        'ffmpeg', '-y',
        '-i', audio_file,
        '-i', temp_click_wav,
        '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]',
        '-map', '[aout]',
        '-c:a', 'aac',
        '-b:a', '320k',
        temp_mix_audio,
    ]
    subprocess.run(mix_cmd, check=True)

    print('Muxing final video...')
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', temp_video,
        '-i', temp_mix_audio,
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-shortest',
        final_output,
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    for path in (temp_video, temp_click_wav, temp_mix_audio):
        try:
            os.remove(path)
        except OSError:
            pass

    print(f'Completed: {final_output}')
