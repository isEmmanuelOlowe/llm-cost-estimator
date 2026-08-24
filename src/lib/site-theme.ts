export const APPLIED_THEMES = ['photonic', 'paper', 'obsidian'] as const;
export type AppliedTheme = (typeof APPLIED_THEMES)[number];

export const THEME_PREFERENCES = [
  'system',
  'paper',
  'obsidian',
  'photonic',
] as const;
export type ThemePreference = (typeof THEME_PREFERENCES)[number];

export const SITE_THEME_STORAGE_KEY = 'labiium:theme';
export const DEFAULT_THEME_PREFERENCE: ThemePreference = 'system';
export const DEFAULT_APPLIED_THEME: AppliedTheme = 'photonic';

export const THEME_LABELS: Record<ThemePreference, string> = {
  system: 'System',
  paper: 'Paper',
  obsidian: 'Obsidian',
  photonic: 'Photonic',
};

export const THEME_BASES: Record<AppliedTheme, string> = {
  photonic: '#0b0e11',
  paper: '#ffffff',
  obsidian: '#191919',
};

export function isThemePreference(value: unknown): value is ThemePreference {
  return (
    typeof value === 'string' &&
    (THEME_PREFERENCES as readonly string[]).includes(value)
  );
}

export function nextThemePreference(current: ThemePreference): ThemePreference {
  const index = THEME_PREFERENCES.indexOf(current);
  return (
    THEME_PREFERENCES[(index + 1) % THEME_PREFERENCES.length] ??
    DEFAULT_THEME_PREFERENCE
  );
}

export function resolveTheme(
  preference: ThemePreference,
  prefersDark: boolean,
): AppliedTheme {
  if (preference === 'system') return prefersDark ? 'obsidian' : 'paper';
  return preference;
}

export function applyTheme(theme: AppliedTheme): void {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  if (theme === DEFAULT_APPLIED_THEME) {
    delete root.dataset.siteTheme;
  } else {
    root.dataset.siteTheme = theme;
  }
  root.style.backgroundColor = THEME_BASES[theme];
}

export function readStoredPreference(): ThemePreference {
  if (typeof window === 'undefined') return DEFAULT_THEME_PREFERENCE;
  try {
    const stored = window.localStorage.getItem(SITE_THEME_STORAGE_KEY);
    return isThemePreference(stored) ? stored : DEFAULT_THEME_PREFERENCE;
  } catch {
    return DEFAULT_THEME_PREFERENCE;
  }
}

export function storePreference(preference: ThemePreference): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(SITE_THEME_STORAGE_KEY, preference);
  } catch {
    // Storage can be unavailable in private browsing or restricted contexts.
  }
}

export function siteThemeBootstrapScript(): string {
  const bases = JSON.stringify(THEME_BASES);
  const key = JSON.stringify(SITE_THEME_STORAGE_KEY);
  return `(function(){try{var b=${bases};var p=localStorage.getItem(${key});if(p!=="paper"&&p!=="obsidian"&&p!=="photonic"){p="system"}var t=p==="system"?(window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches?"obsidian":"paper"):p;var r=document.documentElement;if(t!=="photonic"){r.setAttribute("data-site-theme",t)}else{r.removeAttribute("data-site-theme")}r.style.backgroundColor=b[t]}catch(e){}})()`;
}
