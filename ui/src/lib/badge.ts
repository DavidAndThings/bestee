/** daisyUI colour class per category badge, so each badge type reads as a
 *  distinct colour across the home cards and the chart-setup header. */
const BADGE_CLASS: Record<string, string> = {
  Rotation: "badge-primary",
  Clustering: "badge-secondary",
  Factors: "badge-accent",
  Regimes: "badge-success",
  Reference: "badge-warning",
  Jobs: "badge-info",
  Training: "badge-neutral",
};

/** The daisyUI colour class for a badge label (neutral for any unmapped one). */
export function badgeClass(badge: string): string {
  return BADGE_CLASS[badge] ?? "badge-neutral";
}
