/** Curated hue per category badge, tuned for the dark canvas. Because each
 *  card's action button is kept a different hue from its badge (see
 *  {@link actionButtonClasses}), these names double as the button hue to avoid.
 *  The matching colours live in index.css as `badge-soft-*` / `btn-solid-*`. */
const BADGE_HUE: Record<string, string> = {
  Analysis: "indigo",
  Reference: "amber",
  Monitoring: "emerald",
};

/** Hue name for a badge label (neutral for any unmapped one). */
function badgeHue(badge: string): string {
  return BADGE_HUE[badge] ?? "neutral";
}

/** The soft, tinted badge class for a label (see `badge-soft-*` in index.css). */
export function badgeClass(badge: string): string {
  return `badge-soft-${badgeHue(badge)}`;
}

/** Cohesive cool jewel-tone hues for the solid CTA buttons (see `btn-solid-*`
 *  in index.css), ordered so the rolling assignment reads as a harmonious
 *  spread rather than alternating two colours. */
const BUTTON_HUES = [
  "indigo",
  "sky",
  "violet",
  "teal",
  "blue",
  "cyan",
] as const;

/**
 * Assign every card (given in render order, by its badge label) an action-button
 * colour class such that:
 *  - a card's button hue never matches its own badge hue, and
 *  - no two adjacent cards share a button hue.
 *
 * A rolling cursor walks {@link BUTTON_HUES}, taking the first hue that clears
 * both constraints; with six hues and at most two forbidden per card a choice
 * always exists, so the loop never falls back.
 */
export function actionButtonClasses(badges: (string | undefined)[]): string[] {
  const classes: string[] = [];
  let previous: string | null = null;
  let cursor = 0;
  for (const badge of badges) {
    const ownHue = badge ? badgeHue(badge) : null;
    let chosen: string = BUTTON_HUES[cursor % BUTTON_HUES.length];
    for (let step = 0; step < BUTTON_HUES.length; step += 1) {
      const candidate = BUTTON_HUES[(cursor + step) % BUTTON_HUES.length];
      if (candidate !== previous && candidate !== ownHue) {
        chosen = candidate;
        cursor = (cursor + step + 1) % BUTTON_HUES.length;
        break;
      }
    }
    classes.push(`btn-solid-${chosen}`);
    previous = chosen;
  }
  return classes;
}
