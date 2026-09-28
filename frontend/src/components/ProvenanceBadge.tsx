interface BadgeProps {
  provenance: string
}

const LABELS: Record<string, string> = {
  'real-model': 'Real Model',
  'classical-cv-baseline': 'Classical CV',
  'rule-based': 'Rule-Based',
  'not-configured': 'Not Configured',
}

export default function ProvenanceBadge({ provenance }: BadgeProps) {
  const cls =
    provenance === 'real-model' ? 'badge-real-model' :
    provenance === 'classical-cv-baseline' ? 'badge-classical-cv' :
    provenance === 'rule-based' ? 'badge-rule-based' :
    'badge-not-configured'

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
      {LABELS[provenance] || provenance}
    </span>
  )
}
