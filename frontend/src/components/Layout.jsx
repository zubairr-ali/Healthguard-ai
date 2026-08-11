import { useState } from 'react';
import { NavLink, useLocation, useOutlet } from 'react-router-dom';
import { AnimatePresence, motion, MotionConfig } from 'framer-motion';
import { Sun, Moon, Activity, GitFork } from 'lucide-react';
import { useTheme } from '../hooks/useTheme';
import Waveform from './Waveform';

// A plain <Outlet /> re-reads router context on every render, so if it were
// rendered directly inside the exiting half of an AnimatePresence swap, it
// would immediately switch to the *new* page's content mid fade-out instead
// of animating the old one away. useState's initializer only runs once per
// mount, so this freezes each transition's outlet to the page it started
// with — AnimatePresence gives this instance a fresh mount (and a fresh
// freeze) only when the route key actually changes.
function AnimatedOutlet() {
  const [outlet] = useState(useOutlet());
  return outlet;
}

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/predict/heart', label: 'Heart risk' },
  { to: '/predict/diabetes', label: 'Diabetes risk' },
  { to: '/findings', label: 'Research findings' },
  { to: '/models', label: 'Model comparison' },
  { to: '/history', label: 'History' },
];

function NavItem({ to, label, end }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        [
          'px-3 py-2 text-sm rounded-lg transition-colors font-medium',
          isActive
            ? 'text-vital-400 bg-ink-800/60 light:bg-vital-500/10 light:text-vital-600'
            : 'text-ink-300 hover:text-ink-100 light:text-ink-500 light:hover:text-ink-900',
        ].join(' ')
      }
    >
      {label}
    </NavLink>
  );
}

export default function Layout() {
  const { theme, toggle } = useTheme();
  const location = useLocation();

  return (
    <MotionConfig reducedMotion="user">
      <div className="min-h-screen flex flex-col bg-ink-950 light:bg-paper-50">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:bg-vital-500 focus:text-ink-950 focus:px-4 focus:py-2 focus:rounded-lg"
        >
          Skip to content
        </a>

        <header className="sticky top-0 z-40 backdrop-blur-md bg-ink-950/80 light:bg-paper-50/85 border-b border-ink-800 light:border-ink-100">
          <div className="max-w-7xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between gap-4">
            <NavLink to="/" className="flex items-center gap-2.5 shrink-0" aria-label="HealthGuard AI home">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-vital-500 to-vital-700 flex items-center justify-center">
                <Activity size={17} strokeWidth={2.5} className="text-ink-950" />
              </div>
              <span className="font-display font-semibold text-[15px] tracking-tight text-ink-50 light:text-ink-900">
                HealthGuard<span className="text-vital-400">AI</span>
              </span>
            </NavLink>

            <nav className="hidden md:flex items-center gap-1" aria-label="Primary">
              {NAV.map((n) => (
                <NavItem key={n.to} {...n} />
              ))}
            </nav>

            <div className="flex items-center gap-1.5">
              <a
                href="https://github.com/zubairr-ali/Healthguard-ai"
                target="_blank"
                rel="noreferrer"
                className="p-2 rounded-lg text-ink-400 hover:text-ink-100 hover:bg-ink-800 light:text-ink-400 light:hover:text-ink-900 light:hover:bg-ink-100 transition-colors"
                aria-label="View source on GitHub"
              >
                <GitFork size={18} />
              </a>
              <button
                onClick={toggle}
                className="p-2 rounded-lg text-ink-400 hover:text-ink-100 hover:bg-ink-800 light:text-ink-400 light:hover:text-ink-900 light:hover:bg-ink-100 transition-colors"
                aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
              >
                {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
              </button>
            </div>
          </div>

          <nav className="md:hidden flex items-center gap-1 px-5 pb-2.5 overflow-x-auto" aria-label="Primary">
            {NAV.map((n) => (
              <NavItem key={n.to} {...n} />
            ))}
          </nav>
        </header>

        <main id="main" className="flex-1">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
            >
              <AnimatedOutlet />
            </motion.div>
          </AnimatePresence>
        </main>

        <footer className="border-t border-ink-800 light:border-ink-100 mt-20">
          <div className="max-w-7xl mx-auto px-5 sm:px-8 py-10">
            <Waveform repeats={16} height={28} strokeWidth={1.5} animate={false} glow={false} className="opacity-25 mb-8" />
            <div className="flex flex-col sm:flex-row justify-between gap-3 text-sm text-ink-400 light:text-ink-500">
              <p>
                HealthGuard AI — Computer Science Project, module 6WCM0029, University of Hertfordshire.
              </p>
              <p className="data-readout text-xs text-ink-500 light:text-ink-400">
                Predictions are decision support only. Not a diagnosis. Always consult a clinician.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </MotionConfig>
  );
}
