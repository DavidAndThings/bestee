/** Each category's curated two-tone combo on the dark canvas: a soft, tinted
 *  badge plus a solid jewel-tone CTA button in a different, harmonious hue.
 *  Cards that share a badge therefore share the same badge+button combo, so the
 *  pairing reads as a category. The colours live in index.css as
 *  `badge-soft-*` / `btn-solid-*`. */
const CATEGORY_COMBO: Record<string, { badge: string; button: string }> = {
  // Cool harmony: soft indigo tag, solid deep-sky CTA.
  Analysis: { badge: "indigo", button: "sky" },
  // Warm-on-warm: soft amber tag, solid burnt-orange CTA in the same family.
  Reference: { badge: "amber", button: "orange" },
  // Analogous green family: soft emerald tag, solid teal CTA.
  Monitoring: { badge: "emerald", button: "teal" },
};

/** Fallback combo for any unmapped badge label. */
const NEUTRAL_COMBO = { badge: "neutral", button: "indigo" };

/** The badge+button hue combo for a label (neutral for any unmapped one). */
function combo(badge: string | undefined): { badge: string; button: string } {
  return (badge && CATEGORY_COMBO[badge]) || NEUTRAL_COMBO;
}

/** The soft, tinted badge class for a label (see `badge-soft-*` in index.css). */
export function badgeClass(badge: string): string {
  return `badge-soft-${combo(badge).badge}`;
}

/** The solid CTA button class for a card with the given badge label (see
 *  `btn-solid-*` in index.css). Cards sharing a badge share the same button, so
 *  the combo is consistent per category. */
export function actionButtonClass(badge: string | undefined): string {
  return `btn-solid-${combo(badge).button}`;
}
