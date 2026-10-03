#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import timedelta
from unmanic.libs.unplugins.settings import PluginSettings
from video_transcoder.lib.ffmpeg import Parser, Probe

logger = logging.getLogger("Unmanic.Plugin.ai_subtitle_generator")

# ------------------------------------------------------------------
# Settings class
# ------------------------------------------------------------------
class Settings(PluginSettings):

    settings = {
        "asr_engine": "whisper",
        "whisper_model": "openai/whisper-large-v3",
        "parakeet_model": "nvidia/parakeet-tdt-0.6b-v3",
        "device": "auto",
        "vad_min_silence_ms": 500,
        "vad_speech_pad_ms": 200,
        "overwrite_existing": False,
        "keep_temp_files": False,
        "use_marker_file": True,
    }

    form_settings = {
        "asr_engine": {
            "label": "ASR Engine",
            "input_type": "select",
            "select_options": [
                {"value": "whisper", "label": "Whisper Large v3"},
                {"value": "parakeet", "label": "Parakeet TDT 0.6B v3"},
            ],
        },
        "device": {
            "label": "Compute Device",
            "input_type": "select",
            "select_options": [
                {"value": "auto", "label": "Auto-detect"},
                {"value": "cuda", "label": "CUDA (GPU)"},
                {"value": "cpu", "label": "CPU"},
            ],
        },
    }

    def __init__(self, *args, **kwargs):
        # Add optional flag to disable default fallback logic. This should be set for the plugin runner calls
        self.apply_default_fallbacks = kwargs.pop('apply_default_fallbacks', True)
        super(Settings, self).__init__(*args, **kwargs)
        self.settings = self.settings
        self.form_settings = self.form_settings

def append_worker_log(worker_log, line: str):
    """
    Append a log line to Unmanic's `worker_log` list (if provided).
    """
    if worker_log is None or not isinstance(worker_log, list):
        return
    try:
        worker_log.append(str(line))
    except Exception:
        # Never break processing due to logging issues
        return

def on_library_management_file_test(data, file_metadata=None):
    """
    Queue a video file for subtitle generation IF:
      - it's a recognised video extension
      - it does not already have a matching .srt sidecar
      - (optional) a marker file is not present
    """
    path = data.get('path', '')
    if not path:
        return data

    # Get file probe
    probe = Probe.init_probe(data, logger, allowed_mimetypes=['video'])
    if not probe:
        # File not able to be probed by ffprobe
        return data

    # Load settings so we respect user preferences
    settings = Settings(library_id=data.get('library_id'), apply_default_fallbacks=False)
    overwrite = settings.get_setting('overwrite_existing')
    use_marker = settings.get_setting('use_marker_file')

# Video extensions we care about
    VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.m4v', '.ts', '.webm', '.flv', '.wmv'}
    ext = os.path.splitext(path)[1].lower()
    if ext not in VIDEO_EXTS:
        return data

    srt_path = os.path.splitext(path)[0] + '.srt'
    marker_path = os.path.splitext(path)[0] + '.asr_done'

    worker_log = data.get('worker_log')
    # Already processed?
    if not overwrite and os.path.exists(srt_path):
        append_worker_log(worker_log, f'Skipping (SRT exists): {path}')
        return data

    # Optional marker file (useful if you want to force re-run later by deleting it)
    if use_marker and not overwrite and os.path.exists(marker_path):
        append_worker_log(worker_log, f'Skipping (marker present): {path}')
        return data

    # Queue it
    data['add_file_to_pending_tasks'] = True
    append_worker_log(worker_log, f'Queued for subtitle generation: {path}')
    return data

# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------
def on_worker_process(data, file_metadata=None):
    # Default to no external command required
    data['exec_command'] = []
    data['repeat'] = False

    worker_log = data.get('worker_log')
    source_file = data.get('file_in')
    if not source_file or not Path(source_file).is_file():
        append_worker_log(worker_log, 'Source file not found or invalid: {}'.format(source_file))
        return

    # Load user settings via the Settings class
    settings = Settings(library_id=data.get('library_id'), apply_default_fallbacks=False)

    engine          = settings.get_setting('asr_engine')
    whisper_model   = settings.get_setting('whisper_model')
    parakeet_model  = settings.get_setting('parakeet_model')
    device          = settings.get_setting('device')
    vad_min_silence = int(settings.get_setting('vad_min_silence_ms'))
    vad_speech_pad  = int(settings.get_setting('vad_speech_pad_ms'))
    overwrite       = settings.get_setting('overwrite_existing')
    keep_temp       = settings.get_setting('keep_temp_files')
    use_marker      = settings.get_setting('use_marker_file')

    video_path = Path(source_file)
    srt_path   = video_path.with_suffix('.srt')

    if srt_path.exists() and not overwrite:
        append_worker_log(worker_log, 'SRT already exists and overwrite is disabled: {}'.format(srt_path))
        return

    append_worker_log(worker_log, 'Starting subtitle generation for: {}'.format(source_file))

    tmp_dir = None
    try:
        tmp_dir, clean_audio = _extract_and_clean_audio(
            source_file, vad_min_silence, vad_speech_pad
        )

        if engine == 'parakeet':
            segments = _run_parakeet(clean_audio, parakeet_model, device)
        else:
            segments = _run_whisper(clean_audio, whisper_model, device)

        _write_srt(segments, srt_path)

        if use_marker:
            marker_path = video_path.with_suffix('.asr_done')
            marker_path.touch()
            append_worker_log(worker_log, 'Marker file created: {}'.format(marker_path))

        append_worker_log(worker_log, 'Subtitle generation completed: {}'.format(srt_path))

    except Exception as exc:
        logging.error(f'[ai_subtitle_generator] Subtitle generation failed: {exc}')
        append_worker_log(worker_log, 'Subtitle generation failed: {}'.format(exc))
    finally:
        if tmp_dir and not keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            append_worker_log(worker_log, 'Temporary directory cleaned up: {}'.format(tmp_dir))

    return


# ------------------------------------------------------------------
# Audio extraction + Silero VAD
# ------------------------------------------------------------------
def _extract_and_clean_audio(video_path, min_silence_ms, speech_pad_ms):
    tmp_dir   = tempfile.mkdtemp(prefix='unmanic_asr_')
    raw_wav   = os.path.join(tmp_dir, 'raw.wav')
    clean_wav = os.path.join(tmp_dir, 'clean.wav')

    subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-ac', '1', '-ar', '16000', '-vn', '-c:a', 'pcm_s16le', raw_wav
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    import wave
    import torch

    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]

    with wave.open(raw_wav, 'rb') as audio_file:
        if audio_file.getnchannels() != 1 or audio_file.getsampwidth() != 2 or audio_file.getframerate() != 16000:
            raise ValueError('Extracted audio must be 16 kHz mono PCM16')
        audio_bytes = audio_file.readframes(audio_file.getnframes())
    if not audio_bytes:
        raise ValueError('FFmpeg produced an empty audio file')
    wav = torch.frombuffer(bytearray(audio_bytes), dtype=torch.int16).to(torch.float32).div_(32768.0)

    speech_ts = get_speech_timestamps(
        wav, model, sampling_rate=16000,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )

    if not speech_ts:
        return tmp_dir, raw_wav

    chunks = [wav[ts['start']:ts['end']] for ts in speech_ts]
    clean = torch.cat(chunks)
    pcm = clean.detach().cpu().clamp(-1.0, 1.0).mul(32767).round().to(torch.int16).numpy()
    with wave.open(clean_wav, 'wb') as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(pcm.astype('<i2', copy=False).tobytes())
    return tmp_dir, clean_wav


# ------------------------------------------------------------------
# ASR backends
# ------------------------------------------------------------------
def _resolve_device(device_setting):
    import torch
    if device_setting == 'cuda':
        return 'cuda'
    if device_setting == 'cpu':
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _run_whisper(audio_path, model_id, device_setting):
    import torch
    from transformers import pipeline

    device = _resolve_device(device_setting)
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32

    pipe = pipeline(
        'automatic-speech-recognition',
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe(
        audio_path,
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=True,
    )

    segments = []
    for chunk in result.get('chunks', []):
        start, end = chunk['timestamp']
        segments.append({
            'start': start if start is not None else 0.0,
            'end':   end   if end   is not None else (start or 0.0),
            'text':  chunk['text'].strip(),
        })
    return segments


def _run_parakeet(audio_path, model_id, device_setting):
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    device = _resolve_device(device_setting)
    model = model.cuda() if device == 'cuda' else model.cpu()

    outputs = model.transcribe([audio_path], timestamps=True)

    segments = []
    for hyp in outputs:
        timestep = getattr(hyp, 'timestep', None)
        if timestep and isinstance(timestep, dict):
            entries = timestep.get('word') or timestep.get('segment') or []
            for entry in entries:
                if 'start' in entry and 'end' in entry:
                    segments.append({
                        'start': float(entry['start']),
                        'end':   float(entry['end']),
                        'text':  entry.get('word') or entry.get('segment') or '',
                    })
        if not segments:
            segments.append({'start': 0.0, 'end': 0.0, 'text': hyp.text})
    return segments


# ------------------------------------------------------------------
# SRT writer
# ------------------------------------------------------------------
def _write_srt(segments, srt_path):
    def fmt(seconds):
        td = timedelta(seconds=float(seconds))
        total = int(td.total_seconds())
        ms = int((td.total_seconds() - total) * 1000)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'

    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, start=1):
            text = (seg.get('text') or '').replace('\n', ' ').strip()
            f.write(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{text}\n\n")