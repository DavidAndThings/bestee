/** daisyUI colour *name* per category badge. Kept as a bare name (not a full
 *  class) so a card's action button can be compared against — and made to
 *  differ from — its own badge colour. */
const BADGE_COLOR: Record<string, string> = {
  Analysis: "primary",
  Reference: "warning",
  Monitoring: "info",
};

/** Colour name for a badge label (neutral for any unmapped one). */
function badgeColor(badge: string): string {
  return BADGE_COLOR[badge] ?? "neutral";
}

/** Full daisyUI badge class per colour name. Spelled out as literals (rather
 *  than built as `badge-${name}`) so Tailwind's source scanner keeps emitting
 *  them. */
const BADGE_CLASS: Record<string, string> = {
  primary: "badge-primary",
  warning: "badge-warning",
  info: "badge-info",
  neutral: "badge-neutral",
};

/** The daisyUI colour class for a badge label (neutral for any unmapped one). */
export function badgeClass(badge: string): string {
  return BADGE_CLASS[badgeColor(badge)] ?? "badge-neutral";
}

/** Action-button colours to rotate through for the home cards. Listed as full
 *  literal classes so Tailwind keeps emitting them, and ordered so the rotation
 *  reads as a pleasant spread rather than alternating two hues. `primary` is
 *  included but, being the dominant badge colour, is usually skipped. */
const BUTTON_PALETTE = [
  "btn-secondary",
  "btn-accent",
  "btn-primary",
  "btn-success",
  "btn-error",
  "btn-info",
  "btn-warning",
] as const;

/** The colour name embedded in a `btn-*` palette class. */
function buttonColor(cls: string): string {
  return cls.slice("btn-".length);
}

/**
 * Assign every card (given in render order, by its badge label) an action-button
 * colour class such that:
 *  - a card's button colour never matches its own badge colour, and
 *  - no two adjacent cards share a button colour.
 *
 * A rolling cursor walks {@link BUTTON_PALETTE}, taking the first colour that
 * clears both constraints; with seven colours and at most two forbidden per
 * card a choice always exists, so the loop never falls back.
 */
export function actionButtonClasses(badges: (string | undefined)[]): string[] {
  const classes: string[] = [];
  let previous: string | null = null;
  let cursor = 0;
  for (const badge of badges) {
    const ownColor = badge ? badgeColor(badge) : null;
    let chosen = BUTTON_PALETTE[cursor % BUTTON_PALETTE.length];
    for (let step = 0; step < BUTTON_PALETTE.length; step += 1) {
      const candidate = BUTTON_PALETTE[(cursor + step) % BUTTON_PALETTE.length];
      const color = buttonColor(candidate);
      if (color !== previous && color !== ownColor) {
        chosen = candidate;
        cursor = (cursor + step + 1) % BUTTON_PALETTE.length;
        break;
      }
    }
    classes.push(chosen);
    previous = buttonColor(chosen);
  }
  return classes;
}
