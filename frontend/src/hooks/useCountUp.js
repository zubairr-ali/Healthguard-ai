import { useEffect, useState } from 'react';
import { animate } from 'framer-motion';

export const COUNT_UP_EASE = [0.65, 0, 0.35, 1];

export function useCountUp(target, shouldAnimate) {
  const [value, setValue] = useState(shouldAnimate ? 0 : target);

  useEffect(() => {
    if (!shouldAnimate) {
      setValue(target);
      return;
    }
    const controls = animate(0, target, {
      duration: 1,
      ease: COUNT_UP_EASE,
      onUpdate: setValue,
    });
    return controls.stop;
  }, [target, shouldAnimate]);

  return value;
}
