export interface ModelLicenseInfo {
  id: string;
  label: string;
  summary: string;
  usage: string;
  sourceUrl: string;
  custom: boolean;
}

const LICENSES: Record<string, Omit<ModelLicenseInfo, 'id' | 'custom'>> = {
  'apache-2.0': {
    label: 'Apache 2.0',
    summary: 'Permissive open-source license with an explicit patent grant.',
    usage:
      'Commercial use, modification, and redistribution are generally allowed. Preserve required notices and license text.',
    sourceUrl: 'https://www.apache.org/licenses/LICENSE-2.0',
  },
  mit: {
    label: 'MIT',
    summary: 'Short permissive license with minimal redistribution conditions.',
    usage:
      'Commercial use, modification, sublicensing, and distribution are generally allowed when the copyright and permission notice are retained.',
    sourceUrl: 'https://opensource.org/license/mit',
  },
  'openmdw-1.1': {
    label: 'OpenMDW 1.1',
    summary: 'Open Model, Data and Weights license agreement version 1.1.',
    usage:
      'Commercial deployment is supported, but model, data, attribution, redistribution, and acceptable-use terms should be reviewed for the intended product.',
    sourceUrl:
      'https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/main/LICENSE',
  },
  gemma: {
    label: 'Gemma Terms',
    summary:
      'Google model terms that apply specifically to Gemma distributions.',
    usage:
      'Use and redistribution can be permitted, including commercial use, subject to the Gemma Terms, prohibited-use policy, and downstream notice requirements.',
    sourceUrl: 'https://ai.google.dev/gemma/terms',
  },
  llama: {
    label: 'Llama Community',
    summary:
      'Meta community model license with attribution and use conditions.',
    usage:
      'Commercial use may be allowed, but acceptable-use, attribution, redistribution, and scale-related terms vary by Llama release.',
    sourceUrl: 'https://www.llama.com/llama-downloads/',
  },
  openrail: {
    label: 'OpenRAIL',
    summary: 'Open model license family with behavior-based use restrictions.',
    usage:
      'Use, modification, and distribution may be allowed while specified prohibited-use restrictions remain attached to the model and derivatives.',
    sourceUrl: 'https://huggingface.co/docs/hub/repositories-licenses',
  },
  'kimi-k3': {
    label: 'Kimi K3 License',
    summary: 'Moonshot AI model license with commercial scale conditions.',
    usage:
      'Broad use and modification are allowed, but large Model-as-a-Service businesses and very large commercial products have separate agreement or attribution requirements.',
    sourceUrl: 'https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE',
  },
  'minimax-community': {
    label: 'MiniMax Community',
    summary: 'Community model license with commercial notice and scale terms.',
    usage:
      'Non-commercial use is allowed. Commercial products require attribution/notice, and high-revenue use requires prior written authorization.',
    sourceUrl: 'https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/LICENSE',
  },
  'cc-by-4.0': {
    label: 'CC BY 4.0',
    summary: 'Creative Commons attribution license.',
    usage:
      'Sharing and adaptation, including commercial use, are generally allowed with attribution and indication of changes.',
    sourceUrl: 'https://creativecommons.org/licenses/by/4.0/',
  },
  'bsd-3-clause': {
    label: 'BSD 3-Clause',
    summary: 'Permissive license with notice and non-endorsement conditions.',
    usage:
      'Commercial use, modification, and redistribution are generally allowed when copyright, license, and non-endorsement terms are retained.',
    sourceUrl: 'https://opensource.org/license/bsd-3-clause',
  },
};

const MODEL_LICENSE_OVERRIDES: Array<{
  matches: (modelId: string) => boolean;
  licenseId: string;
}> = [
  {
    matches: (modelId) =>
      modelId === 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
    licenseId: 'openmdw-1.1',
  },
  {
    matches: (modelId) => modelId === 'moonshotai/Kimi-K3',
    licenseId: 'kimi-k3',
  },
  {
    matches: (modelId) => modelId === 'MiniMaxAI/MiniMax-M3',
    licenseId: 'minimax-community',
  },
];

function normalizeLicenseId(license: string | null | undefined): string {
  const normalized = license?.trim().toLowerCase() ?? '';
  if (/^llama(?:2|3|3\.1|3\.2|3\.3|4)$/.test(normalized)) return 'llama';
  if (normalized.includes('openrail')) return 'openrail';
  if (normalized === 'bsd' || normalized === 'bsd-3') return 'bsd-3-clause';
  return normalized;
}

export function resolveModelLicense({
  modelId,
  license,
  modelUrl,
}: {
  modelId: string;
  license?: string | null;
  modelUrl: string;
}): ModelLicenseInfo {
  const override = MODEL_LICENSE_OVERRIDES.find((entry) =>
    entry.matches(modelId),
  );
  const id = override?.licenseId ?? normalizeLicenseId(license);
  const known = LICENSES[id];
  if (known) return { id, ...known, custom: false };

  return {
    id: id || 'unknown',
    label: id && id !== 'other' ? (license ?? id) : 'Model-specific license',
    summary:
      'The repository uses terms that are not represented by a common SPDX-style identifier.',
    usage:
      'Review the model card and full license before commercial use, redistribution, fine-tuning, hosting, or publishing derivatives.',
    sourceUrl: `${modelUrl.replace(/\/$/, '')}/blob/main/LICENSE`,
    custom: true,
  };
}
