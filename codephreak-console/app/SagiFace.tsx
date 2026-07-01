'use client';
// sAGI's sensorium — a face for the sandbox intelligence:
//   • mouth  → an oscilloscope, driven by real audio (Web Audio AnalyserNode) and
//              by local-CPU TTS (window.speechSynthesis) when sAGI speaks;
//   • ears   → microphone audio-in; the two ear dots pulse with the input level;
//   • eyes   → two live video feeds (camera + webcam).
// All local: getUserMedia + on-device speech synthesis, nothing leaves the machine.
// Gated behind an explicit Activate button (camera/mic need user permission).
import { useEffect, useRef, useState } from 'react';

export default function SagiFace({ speakText }: { speakText?: string }) {
  const [active, setActive] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [err, setErr] = useState('');
  const [level, setLevel] = useState(0);
  const mouthRef = useRef<HTMLCanvasElement>(null);
  const eye1 = useRef<HTMLVideoElement>(null);
  const eye2 = useRef<HTMLVideoElement>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamsRef = useRef<MediaStream[]>([]);
  const speakingRef = useRef(false); speakingRef.current = speaking;

  async function activate() {
    setErr('');
    try {
      const mic = await navigator.mediaDevices.getUserMedia({ audio: true });   // ears
      streamsRef.current.push(mic);
      const AC = (window.AudioContext || (window as any).webkitAudioContext);
      const ctx = new AC(); ctxRef.current = ctx;
      const an = ctx.createAnalyser(); an.fftSize = 1024;
      ctx.createMediaStreamSource(mic).connect(an);
      analyserRef.current = an;

      // eyes — grant video, then split across two cameras if available.
      const s1 = await navigator.mediaDevices.getUserMedia({ video: true });
      streamsRef.current.push(s1);
      if (eye1.current) eye1.current.srcObject = s1;
      let s2: MediaStream = s1;
      try {
        const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
        if (cams.length > 1) { s2 = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: cams[1].deviceId } } }); streamsRef.current.push(s2); }
      } catch { /* one camera only */ }
      if (eye2.current) eye2.current.srcObject = s2;
      setActive(true);
    } catch (e: any) {
      setErr(e?.name === 'NotAllowedError' ? 'permission denied — allow camera/mic to give sAGI its senses' : (e?.message || 'could not access devices'));
    }
  }

  function deactivate() {
    streamsRef.current.forEach((s) => s.getTracks().forEach((t) => t.stop()));
    streamsRef.current = [];
    try { ctxRef.current?.close(); } catch { /* noop */ }
    ctxRef.current = null; analyserRef.current = null;
    setActive(false);
  }

  function speak(text?: string) {
    if (!text || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    const u = new SpeechSynthesisUtterance(text.slice(0, 320));
    u.rate = 1; u.pitch = 1;
    u.onstart = () => setSpeaking(true);
    u.onend = () => setSpeaking(false);
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }

  // Oscilloscope mouth + ear level.
  useEffect(() => {
    if (!active || !mouthRef.current) return;
    const cv = mouthRef.current; const g = cv.getContext('2d')!;
    const an = analyserRef.current; const buf = an ? new Uint8Array(an.fftSize) : null;
    let raf = 0, t = 0, lvl = 0;
    const draw = () => {
      t += 0.05;
      const w = cv.width = cv.clientWidth * 2, h = cv.height = cv.clientHeight * 2;
      g.clearRect(0, 0, w, h);
      g.lineWidth = 2.4; g.strokeStyle = speakingRef.current ? '#2ee6a6' : '#37b6ff';
      g.shadowBlur = 14; g.shadowColor = g.strokeStyle;
      g.beginPath();
      if (an && buf) an.getByteTimeDomainData(buf);
      let sum = 0;
      for (let x = 0; x < w; x++) {
        let v: number;
        if (an && buf) { v = (buf[Math.floor((x / w) * buf.length)] - 128) / 128; }
        else { v = speakingRef.current ? Math.sin(x * 0.05 + t * 3) * (0.35 + 0.3 * Math.sin(t * 6)) * Math.sin(x * 0.25) : Math.sin(x * 0.02 + t) * 0.03; }
        if (speakingRef.current) v *= 1.5;
        sum += Math.abs(v);
        const y = h / 2 + v * h * 0.42;
        x === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
      }
      g.stroke();
      lvl += (Math.min(1, (sum / w) * 5) - lvl) * 0.25;
      setLevel(lvl);
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [active]);

  useEffect(() => () => deactivate(), []);   // stop devices on unmount

  const earOn = level > 0.06;
  return (
    <div className="sagiface">
      <div className="sf-eyes">
        <div className="sf-eye"><video ref={eye1} autoPlay muted playsInline /><span className="sf-cap">👁 camera</span></div>
        <div className="sf-eye"><video ref={eye2} autoPlay muted playsInline /><span className="sf-cap">👁 webcam</span></div>
      </div>
      <div className="sf-mouth">
        <span className={'sf-ear' + (earOn ? ' on' : '')} title="left ear — audio in" style={{ transform: `scale(${1 + level * 0.6})` }} />
        <canvas ref={mouthRef} className="sf-scope" />
        <span className={'sf-ear' + (earOn ? ' on' : '')} title="right ear — audio in" style={{ transform: `scale(${1 + level * 0.6})` }} />
      </div>
      <div className="sf-ctl">
        {!active
          ? <button className="btn sm primary" onClick={activate}>◉ Activate sAGI senses</button>
          : <>
              <span className="sf-badge">👂 ears · 👁 eyes live</span>
              <button className="btn sm" onClick={() => speak(speakText)} disabled={!speakText}>{speaking ? '🗣 speaking…' : '🔊 speak'}</button>
              <button className="btn ghost sm" onClick={deactivate}>senses off</button>
            </>}
        {err && <span className="sf-err">{err}</span>}
      </div>
    </div>
  );
}
