
import { useReducedMotion } from 'framer-motion';
import { useCountUp } from '../hooks/useCountUp';

function bandFor(p) {
  if (p < 0.3) return { label: 'Low risk', color: 'var(--color-vital-400)', className: 'text-vital-400 light:text-[#0d7c70]' };
  if (p < 0.6) return { label: 'Moderate risk', color: 'var(--color-amber-500)', className: 'text-amber-500 light:text-[#96631b]' };
  return { label: 'High risk', color: 'var(--color-signal-500)', className: 'text-signal-500 light:text-[#d72413]' };
}

export default function RiskGauge({ probability, size = 180 }) {
  const p = Math.max(0, Math.min(1, Number.isFinite(probability) ? probability : 0));
  const band = bandFor(p);
  const r = (size - 20) / 2;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - p);
  const prefersReducedMotion = useReducedMotion();
  const displayPct = useCountUp(p * 100, !prefersReducedMotion);

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90">
          <circle
            cx={size / 2} cy={size / 2} r={r}
            fill="none" strokeWidth="12"
            className="stroke-ink-800 light:stroke-ink-100"
          />
          <circle
            cx={size / 2} cy={size / 2} r={r}
            fill="none" strokeWidth="12" strokeLinecap="round"
            stroke={band.color}
            strokeDasharray={c}
            strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 1s cubic-bezier(0.65,0,0.35,1), stroke 0.4s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="data-readout text-4xl font-semibold text-ink-50 light:text-ink-900">
            {displayPct.toFixed(1)}
          </span>
          <span className="data-readout text-xs text-ink-400 light:text-ink-500 -mt-1">percent</span>
        </div>
      </div>
      <span className={`text-sm font-semibold ${band.className}`}>{band.label}</span>
    </div>
  );
}
