/**
 * The project's signature visual element. HealthGuard AI's own research
 * (Phase 2) is ECG classification, so the waveform trace is not decoration
 * borrowed from generic medical branding — it is the artefact the project
 * actually produces. Reused as: hero centerpiece, section divider, and
 * (in compact form) a loading indicator.
 *
 * The path approximates a real single PQRST cycle repeated, not an
 * arbitrary squiggle, drawn with a self-animating stroke.
 */

const BEAT_SEGMENT = [
  [0, 40], [28, 40], [34, 34], [40, 46], [46, 20], [52, 58], [58, 40], [64, 40],
];

function buildPath(repeats) {
  let d = '';
  for (let i = 0; i < repeats; i++) {
    const offset = i * 64;
    BEAT_SEGMENT.forEach(([x, y], idx) => {
      const cmd = i === 0 && idx === 0 ? 'M' : 'L';
      d += `${cmd}${x + offset},${y} `;
    });
  }
  return d.trim();
}

export default function Waveform({
  repeats = 10,
  height = 80,
  className = '',
  strokeWidth = 2.5,
  color = 'var(--color-vital-400)',
  animate = true,
  glow = true,
  loop = false,
}) {
  const width = repeats * 64;
  const d = buildPath(repeats);
  const len = repeats * 130;

  return (
    <svg
      viewBox={`0 10 ${width} 60`}
      width="100%"
      height={height}
      preserveAspectRatio="none"
      className={className}
      aria-hidden="true"
    >
      {glow && (
        <path
          d={d}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth * 3}
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity="0.15"
          style={{ filter: 'blur(4px)' }}
        />
      )}
      <path
        d={d}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        className={animate ? (loop ? 'waveform-trace-loop' : 'waveform-trace') : ''}
        style={animate ? { '--trace-len': len } : undefined}
      />
    </svg>
  );
}

/** Small inline pulse used inside buttons/badges while a request is in flight. */
export function WaveformPulse({ size = 20, color = 'currentColor' }) {
  return (
    <svg viewBox="0 0 60 20" width={size * 3} height={size} aria-hidden="true">
      <path
        d="M0,10 L14,10 L17,5 L20,15 L23,2 L26,18 L29,10 L60,10"
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="animate-pulse-soft"
      />
    </svg>
  );
}
