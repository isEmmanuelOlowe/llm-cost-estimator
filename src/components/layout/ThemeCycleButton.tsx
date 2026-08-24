import { useEffect, useState } from 'react';

import {
  applyTheme,
  DEFAULT_THEME_PREFERENCE,
  nextThemePreference,
  readStoredPreference,
  resolveTheme,
  storePreference,
  THEME_LABELS,
  type ThemePreference,
} from '@/lib/site-theme';

const GLYPHS: Record<ThemePreference, string> = {
  system: '◐',
  paper: '○',
  obsidian: '●',
  photonic: '◉',
};

export default function ThemeCycleButton() {
  const [preference, setPreference] = useState<ThemePreference>(
    DEFAULT_THEME_PREFERENCE,
  );
  const [preferenceLoaded, setPreferenceLoaded] = useState(false);
  const [prefersDark, setPrefersDark] = useState(false);

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const query = window.matchMedia('(prefers-color-scheme: dark)');
    const sync = () => setPrefersDark(query.matches);
    sync();
    query.addEventListener('change', sync);
    return () => query.removeEventListener('change', sync);
  }, []);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      setPreference(readStoredPreference());
      setPreferenceLoaded(true);
    });
    return () => window.cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    if (!preferenceLoaded || preference !== 'system') return;
    applyTheme(resolveTheme('system', prefersDark));
  }, [preference, preferenceLoaded, prefersDark]);

  const upcoming = nextThemePreference(preference);
  const applied = resolveTheme(preference, prefersDark);
  const label =
    preference === 'system'
      ? `${THEME_LABELS.system} (${THEME_LABELS[applied]})`
      : THEME_LABELS[preference];

  return (
    <button
      type='button'
      onClick={() => {
        setPreference(upcoming);
        storePreference(upcoming);
        applyTheme(resolveTheme(upcoming, prefersDark));
      }}
      title={`Theme: ${label} — switch to ${THEME_LABELS[upcoming]}`}
      aria-label={`Theme: ${label}. Switch to ${THEME_LABELS[upcoming]}.`}
      className='theme-cycle-button inline-flex shrink-0 items-center gap-2 rounded-full px-3 py-1.5 text-xs'
    >
      <span aria-hidden='true' className='text-[11px] leading-none'>
        {GLYPHS[preference]}
      </span>
      <span className='hidden sm:inline'>{THEME_LABELS[preference]}</span>
    </button>
  );
}
