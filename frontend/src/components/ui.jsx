import { ShieldCheck } from 'lucide-react';
import { motion } from 'framer-motion';

export function Card({ children, className = '' }) {
  return (
    <motion.div
      whileHover={{ y: -4 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      className={`rounded-2xl border border-ink-800 light:border-ink-100 bg-ink-900/60 light:bg-white shadow-[0_1px_0_0_rgba(255,255,255,0.03)_inset] ${className}`}
    >
      {children}
    </motion.div>
  );
}

export function Field({ label, hint, error, children, htmlFor }) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className="block text-sm font-medium text-ink-200 light:text-ink-700">
        {label}
      </label>
      {children}
      {hint && !error && <p className="text-xs text-ink-500 light:text-ink-400">{hint}</p>}
      {error && <p className="text-xs text-signal-400">{error}</p>}
    </div>
  );
}

const inputBase =
  'w-full rounded-lg border bg-ink-950/60 light:bg-paper-50 px-3.5 py-2.5 text-sm text-ink-50 light:text-ink-900 ' +
  'placeholder:text-ink-500 transition-colors focus:outline-none focus:ring-2 focus:ring-vital-400/50 ' +
  'focus:border-vital-500';

export function NumberInput({ error, className = '', ...props }) {
  return (
    <input
      type="number"
      inputMode="decimal"
      className={`${inputBase} data-readout ${
        error ? 'border-signal-500' : 'border-ink-700 light:border-ink-200'
      } ${className}`}
      {...props}
    />
  );
}

export function Select({ error, className = '', children, ...props }) {
  return (
    <select
      className={`${inputBase} ${error ? 'border-signal-500' : 'border-ink-700 light:border-ink-200'} ${className}`}
      {...props}
    >
      {children}
    </select>
  );
}

export function Badge({ children, tone = 'neutral', className = '' }) {
  const tones = {
    neutral: 'bg-ink-800 text-ink-300 light:bg-ink-100 light:text-ink-600',
    vital: 'bg-vital-500/10 text-vital-400 border border-vital-500/25',
    signal: 'bg-signal-500/10 text-signal-400 border border-signal-500/25',
    amber: 'bg-amber-500/10 text-amber-400 border border-amber-500/25',
  };
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${tones[tone]} ${className}`}>
      {children}
    </span>
  );
}

export function ModelBadge({ label, name, metric, tone = 'vital' }) {
  return (
    <div className="flex items-center gap-2.5 flex-wrap">
      <Badge tone={tone}>
        <ShieldCheck size={12} className="shrink-0" />
        {label}
      </Badge>
      <span className="text-xs text-ink-500 light:text-ink-400 data-readout">
        {name} · {metric}
      </span>
    </div>
  );
}

export function Button({ children, variant = 'primary', className = '', ...props }) {
  const variants = {
    primary:
      'bg-vital-500 text-ink-950 hover:bg-vital-400 shadow-[0_0_0_1px_rgba(20,184,166,0.3),0_8px_20px_-6px_rgba(20,184,166,0.5)]',
    ghost:
      'bg-transparent text-ink-200 light:text-ink-700 border border-ink-700 light:border-ink-200 hover:bg-ink-800 light:hover:bg-ink-50',
  };
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-5 py-2.5 text-sm font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
