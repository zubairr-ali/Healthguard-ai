import { useEffect, useState } from 'react';

// `ids` must be a stable array reference (a module-level constant, not an
// inline literal) — it's used directly as an effect dependency.
export function useScrollSpy(ids, rootMargin = '-96px 0px -70% 0px') {
  const [activeId, setActiveId] = useState(ids[0] ?? null);

  useEffect(() => {
    const elements = ids.map((id) => document.getElementById(id)).filter(Boolean);
    if (!elements.length) return undefined;

    // Among sections currently in the "active band" (below the sticky
    // header, above the bottom 70% of the viewport), highlight whichever
    // one is closest to the top — that's the one the reader is actually
    // reading, not just the first one mounted.
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (!visible.length) return;
        const top = visible.reduce((a, b) => (a.boundingClientRect.top < b.boundingClientRect.top ? a : b));
        setActiveId(top.target.id);
      },
      { rootMargin, threshold: 0 }
    );

    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [ids, rootMargin]);

  return activeId;
}
