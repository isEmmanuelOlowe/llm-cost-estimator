import {
  isThemePreference,
  nextThemePreference,
  resolveTheme,
} from '../site-theme';

describe('site theme', () => {
  it('cycles through the same theme order as labiium_web', () => {
    expect(nextThemePreference('system')).toBe('paper');
    expect(nextThemePreference('paper')).toBe('obsidian');
    expect(nextThemePreference('obsidian')).toBe('photonic');
    expect(nextThemePreference('photonic')).toBe('system');
  });

  it('resolves system preference without accepting unknown stored values', () => {
    expect(resolveTheme('system', false)).toBe('paper');
    expect(resolveTheme('system', true)).toBe('obsidian');
    expect(isThemePreference('photonic')).toBe(true);
    expect(isThemePreference('midnight')).toBe(false);
  });
});
