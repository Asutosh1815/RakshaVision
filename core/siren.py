"""
RakshaVision AI - Industrial Audio Siren & Emergency Deluge Module
Synthesizes authentic industrial emergency wail sirens in pure Python (no external sound files required)
and renders cross-browser HTML5 Web Audio and embedded base64 WAV playback for Streamlit.
"""

import math
import struct
import io
import wave
import base64
from functools import lru_cache
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components


@lru_cache(maxsize=4)
def generate_siren_wav(duration_sec: float = 2.0, sample_rate: int = 22050) -> bytes:
    """
    Generates a 16-bit PCM Mono WAV byte stream of a dual-tone industrial emergency siren.
    Alternates frequency smoothly between 600 Hz and 960 Hz with urgent harmonic sweep.
    """
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)

        total_samples = int(duration_sec * sample_rate)
        # Modulate frequency at 2.5 cycles per second
        mod_freq = 2.5
        center_freq = 780.0
        depth = 220.0

        for i in range(total_samples):
            t = float(i) / sample_rate
            # Dynamic frequency modulation (wailing warble)
            instant_freq = center_freq + depth * math.sin(2.0 * math.pi * mod_freq * t)
            phase = 2.0 * math.pi * instant_freq * t

            # Primary tone + secondary harmonic for rich industrial penetration
            val1 = math.sin(phase)
            val2 = 0.35 * math.sin(2.0 * phase)
            val = (val1 + val2) / 1.35

            # Volume envelope with smooth decay at the end
            if i > total_samples - 800:
                decay = float(total_samples - i) / 800.0
                val *= decay

            sample = int(0.75 * 32767.0 * val)
            clamped = max(-32767, min(32767, sample))
            wf.writeframes(struct.pack('<h', clamped))

    return buf.getvalue()


@lru_cache(maxsize=4)
def get_siren_base64(duration_sec: float = 2.0) -> str:
    """Returns base64-encoded string of the synthesized siren WAV."""
    wav_bytes = generate_siren_wav(duration_sec=duration_sec)
    return base64.b64encode(wav_bytes).decode('ascii')


def render_siren_audio(continuous: bool = False, key: Optional[str] = None):
    """
    Executes cross-browser siren audio inside a Streamlit HTML iframe.
    Combines embedded Base64 HTML5 Audio with Web Audio API synthesizer for guaranteed playback.
    """
    b64_audio = get_siren_base64(duration_sec=2.2)
    loop_attr = "loop" if continuous else ""

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; }}
      </style>
    </head>
    <body>
      <audio id="rvSirenAudio" autoplay {loop_attr} preload="auto">
        <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
      </audio>

      <script>
        (function() {{
          // 1. Try native audio element play
          var audio = document.getElementById('rvSirenAudio');
          if (audio) {{
            audio.volume = 0.95;
            var playPromise = audio.play();
            if (playPromise !== undefined) {{
              playPromise.catch(function(error) {{
                // Autoplay was blocked; fallback to Web Audio API
                playWebAudioSiren();
              }});
            }}
          }}

          // 2. Web Audio API synthesized backup
          function playWebAudioSiren() {{
            try {{
              var AudioCtx = window.AudioContext || window.webkitAudioContext;
              if (!AudioCtx) return;
              var ctx = new AudioCtx();
              if (ctx.state === 'suspended') {{
                ctx.resume();
              }}

              var osc = ctx.createOscillator();
              var gain = ctx.createGain();
              osc.type = 'sawtooth';

              var now = ctx.currentTime;
              osc.frequency.setValueAtTime(650, now);
              osc.frequency.linearRampToValueAtTime(950, now + 0.30);
              osc.frequency.linearRampToValueAtTime(650, now + 0.60);
              osc.frequency.linearRampToValueAtTime(950, now + 0.90);
              osc.frequency.linearRampToValueAtTime(650, now + 1.20);

              gain.gain.setValueAtTime(0.4, now);
              gain.gain.linearRampToValueAtTime(0.01, now + 1.25);

              osc.connect(gain);
              gain.connect(ctx.destination);
              osc.start(now);
              osc.stop(now + 1.30);
            }} catch(e) {{
              console.warn("Web Audio fallback error:", e);
            }}
          }}
        }})();
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=0, width=0)


def play_siren_audio(continuous: bool = False):
    """
    Renders cross-browser industrial emergency wail siren audio
    using native Streamlit audio with autoplay AND Web Audio API fallback.
    """
    try:
        wav_bytes = generate_siren_wav(duration_sec=2.2 if not continuous else 3.5)
        # 1. Native Streamlit audio player with autoplay
        st.audio(wav_bytes, format="audio/wav", autoplay=True)
    except Exception:
        pass
    # 2. Cross-browser Web Audio API iframe trigger
    try:
        render_siren_audio(continuous=continuous)
    except Exception:
        pass
