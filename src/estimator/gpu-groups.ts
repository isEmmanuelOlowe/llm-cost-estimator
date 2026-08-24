export interface GpuGroupRecord {
  name: string;
  vendor?: string | null;
  architecture?: string | null;
  status?: string | null;
}

export interface GpuGroup<T extends GpuGroupRecord> {
  key: string;
  label: string;
  vendor: string;
  architecture: string;
  gpus: T[];
}

const vendorOrder = ['NVIDIA', 'AMD', 'Intel', 'Apple'];
const statusOrder: Record<string, number> = {
  current: 0,
  'derived-system': 1,
  reference: 2,
  legacy: 3,
};

function orderedValue(value: string, order: string[]): number {
  const index = order.indexOf(value);
  return index === -1 ? order.length : index;
}

export function groupGpus<T extends GpuGroupRecord>(gpus: T[]): GpuGroup<T>[] {
  const groups = new Map<string, GpuGroup<T>>();

  gpus.forEach((gpu) => {
    const vendor = gpu.vendor?.trim() || 'Other vendors';
    const architecture = gpu.architecture?.trim() || 'Other architectures';
    const groupArchitecture =
      vendor === 'Apple' ? 'Unified memory' : architecture;
    const key = `${vendor}:${groupArchitecture}`;
    const group = groups.get(key);
    if (group) {
      group.gpus.push(gpu);
      return;
    }
    groups.set(key, {
      key,
      label: `${vendor} · ${groupArchitecture}`,
      vendor,
      architecture: groupArchitecture,
      gpus: [gpu],
    });
  });

  return [...groups.values()]
    .map((group) => ({
      ...group,
      gpus: [...group.gpus].sort((left, right) => {
        const statusDifference =
          (statusOrder[left.status ?? ''] ?? 4) -
          (statusOrder[right.status ?? ''] ?? 4);
        return statusDifference || left.name.localeCompare(right.name);
      }),
    }))
    .sort((left, right) => {
      const vendorDifference =
        orderedValue(left.vendor, vendorOrder) -
        orderedValue(right.vendor, vendorOrder);
      return (
        vendorDifference ||
        left.vendor.localeCompare(right.vendor) ||
        left.architecture.localeCompare(right.architecture)
      );
    });
}
