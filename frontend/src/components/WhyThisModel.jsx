import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ChevronDown, ArrowRight } from 'lucide-react';

export default function WhyThisModel({ children }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex items-center gap-1.5 text-xs font-medium text-ink-400 light:text-ink-500 hover:text-vital-400 transition-colors"
      >
        <ChevronDown size={13} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
        Why this model?
      </button>
      {open && (
        <div className="mt-2.5 p-4 rounded-lg bg-ink-900/40 light:bg-ink-50 border border-ink-800 light:border-ink-100 max-w-md">
          <p className="text-xs text-ink-300 light:text-ink-600 leading-relaxed">{children}</p>
          <Link
            to="/models"
            className="inline-flex items-center gap-1 text-xs font-medium text-vital-400 hover:gap-1.5 transition-all mt-2.5"
          >
            See the full comparison <ArrowRight size={11} />
          </Link>
        </div>
      )}
    </div>
  );
}
